#include <projects/gaussianviewer/renderer/ImageConvert.hpp>

torch::Tensor sibr::convert_to_float_tensor(const cv::Mat &image)
{

	cv::Mat rgb_image;
	if (image.channels() == 1)
	{
		cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
	}
	else if (image.channels() == 3)
	{
		rgb_image = image.clone();
	}
	else
	{
		throw std::runtime_error("Unsupported number of channels");
	}

	torch::Tensor tensor = torch::from_blob(rgb_image.data, {rgb_image.rows, rgb_image.cols, 3}, torch::kByte).clone();

	return tensor.permute({2, 0, 1}).to(torch::kFloat32).div_(255.0).contiguous();
}

torch::Tensor sibr::normalize(const torch::Tensor &input)
{
	if (input.numel() == 0)
		return input;
	auto min_val = torch::min(input);
	auto max_val = torch::max(input);
	if (max_val.item<float>() == min_val.item<float>())
	{
		return torch::zeros_like(input);
	}
	return (input - min_val) / (max_val - min_val);
}

torch::Tensor sibr::vis_depth(const torch::Tensor &depth_tensor)
{
	torch::Tensor inverted_depth = 1.0f / depth_tensor.clamp(1.0f, 50.0f);

	inverted_depth = inverted_depth.squeeze(0).detach().cpu();
	cv::Mat depth_mat(inverted_depth.size(0), inverted_depth.size(1), CV_32FC1, inverted_depth.data_ptr<float>());

	cv::Mat normalized, normalized_8u;
	normalized = 255.0 * (1.0f - depth_mat);
	normalized.convertTo(normalized_8u, CV_8UC1);

	cv::Mat colored;
	cv::applyColorMap(normalized_8u, colored, cv::COLORMAP_VIRIDIS);

	cv::cvtColor(colored, colored, cv::COLOR_BGR2RGB);
	return convert_to_float_tensor(colored);
}

torch::Tensor sibr::vis_normal(const torch::Tensor &normal_tensor)
{
	TORCH_CHECK(normal_tensor.size(0) == 3, "Input tensor must have shape 3xHxW. Got: ", normal_tensor.sizes());

	auto normal = normal_tensor.detach().clamp(-1.0, 1.0).add(1.0).div(2.0);

	auto mask = torch::isnan(normal) | torch::isinf(normal);
	normal.masked_fill_(mask, 0);

	return normal.contiguous();
}

torch::Tensor sibr::vis_seg(const torch::Tensor &seg_tensor)
{
	torch::Tensor seg_index = seg_tensor.detach().cpu();

	if (seg_index.dim() == 3)
	{
		seg_index = seg_index.squeeze(0);
	}

	const int H = seg_index.size(0);
	const int W = seg_index.size(1);

	cv::Mat colored_img(H, W, CV_8UC3, cv::Scalar(0, 0, 0));

	auto index_accessor = seg_index.accessor<long, 2>();
	for (int y = 0; y < H; ++y)
	{
		for (int x = 0; x < W; ++x)
		{
			int class_idx = index_accessor[y][x];

			if (class_idx < 0)
				class_idx = 0;
			if (class_idx >= static_cast<int>(seg_class_colors.size()))
			{
				class_idx = seg_class_colors.size() - 1;
			}

			colored_img.at<cv::Vec3b>(y, x) = seg_class_colors[class_idx];
		}
	}

	return convert_to_float_tensor(colored_img);
}

void sibr::print_tensor_info(const torch::Tensor &tensor)
{
	std::cout << "size: " << tensor.sizes();
	std::cout << "min " << tensor.min();
	std::cout << "max " << tensor.max();
	std::cout << "mean " << tensor.mean()
			  << std::endl;
}

void sibr::test(
	float *rgb_buffer,
	int H, int W,
	float *rgb_image_out)
{

	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto rgb_image = torch::from_blob(rgb_buffer, {3, H, W}, options);

	rgb_image = rgb_image.contiguous();

	CUDA_SAFE_CALL(cudaMemcpy(rgb_image_out, rgb_image.data_ptr<float>(),
							  3 * H * W * sizeof(float), cudaMemcpyDeviceToDevice));
}

void sibr::RasterizeGaussiansConvertImageCUDA(
	// input
	float *depth_buffer,
	float *normal_buffer, int C_normal,
	float *seg_buffer, int C_seg,
	float *knnmap_buffer,
	int H, int W,
	// output
	float *depth_image_out,
	float *normal_image_out,
	float *seg_image_out,
	float *knnmap_image_out)
{
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

	auto depth_tensor = torch::from_blob(depth_buffer, {H, W}, options);
	auto normal_tensor = torch::from_blob(normal_buffer, {C_normal, H, W}, options);
	auto seg_tensor = torch::from_blob(seg_buffer, {C_seg, H, W}, options);
	auto knnmap_tensor = torch::from_blob(knnmap_buffer, {H, W}, options);

	auto [depth_image, normal_image, seg_image, knnmap_image] =
		OriginalRasterizeGaussiansConvertImageCUDA(
			depth_tensor,
			normal_tensor,
			seg_tensor,
			knnmap_tensor);

	depth_image = depth_image.contiguous();
	normal_image = normal_image.contiguous();
	seg_image = seg_image.contiguous();
	knnmap_image = knnmap_image.contiguous();

	CUDA_SAFE_CALL(cudaMemcpy(depth_image_out, depth_image.data_ptr<float>(),
							  3 * H * W * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpy(normal_image_out, normal_image.data_ptr<float>(),
							  3 * H * W * sizeof(float), cudaMemcpyDeviceToDevice));

	CUDA_SAFE_CALL(cudaMemcpy(seg_image_out, seg_image.data_ptr<float>(),
							  3 * H * W * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpy(knnmap_image_out, knnmap_image.data_ptr<float>(),
							  3 * H * W * sizeof(float), cudaMemcpyDeviceToDevice));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sibr::OriginalRasterizeGaussiansConvertImageCUDA(
	const torch::Tensor &depth_buffer,
	const torch::Tensor &normal_buffer,
	const torch::Tensor &seg_buffer,
	const torch::Tensor &knnmap_buffer)
{

	torch::Tensor normal_image = vis_normal(normal_buffer);

	torch::Tensor seg_argmax = torch::argmax(seg_buffer, 0, true);
	torch::Tensor seg_image = vis_seg(seg_argmax);

	torch::Tensor depth_image = vis_depth(depth_buffer);

	torch::Tensor knnmap_image = normalize(knnmap_buffer)
									 .repeat({3, 1, 1})
									 .contiguous();

	return std::make_tuple(depth_image, normal_image, seg_image, knnmap_image);
}