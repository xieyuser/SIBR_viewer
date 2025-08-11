#pragma once
#include <torch/torch.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define CUDA_SAFE_CALL_ALWAYS(A)              \
	A;                                        \
	cudaDeviceSynchronize();                  \
	if (cudaPeekAtLastError() != cudaSuccess) \
		SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
#define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
#define CUDA_SAFE_CALL(A) A
#endif

namespace sibr
{

#define CHECK_CUDA_ERROR(err)                                                                          \
	do                                                                                                 \
	{                                                                                                  \
		if (err != cudaSuccess)                                                                        \
		{                                                                                              \
			fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(EXIT_FAILURE);                                                                        \
		}                                                                                              \
	} while (0)

	using Vec3b = cv::Vec<uchar, 3>;

	const std::vector<cv::Vec3b> seg_class_colors = {
		cv::Vec3b(0, 0, 0),
		cv::Vec3b(0, 113, 188),
		cv::Vec3b(76, 76, 76),
		cv::Vec3b(170, 84, 127),
		cv::Vec3b(0, 0, 84),
		cv::Vec3b(255, 255, 127),
		cv::Vec3b(170, 0, 127),
		cv::Vec3b(255, 84, 127),
		cv::Vec3b(255, 84, 0),
		cv::Vec3b(84, 170, 127),
		cv::Vec3b(0, 42, 0),
		cv::Vec3b(84, 0, 255),
		cv::Vec3b(255, 84, 0),
		cv::Vec3b(84, 0, 127),
		cv::Vec3b(170, 0, 0),
	};

	torch::Tensor normalize(const torch::Tensor &input);
	torch::Tensor vis_depth(const torch::Tensor &depth_tensor);
	torch::Tensor vis_normal(const torch::Tensor &normal_tensor);
	cv::Mat process_input(const torch::Tensor &img);
	torch::Tensor vis_seg(const torch::Tensor &seg_tensor);

	torch::Tensor convert_to_float_tensor(const cv::Mat &image);

	void RasterizeGaussiansConvertImageCUDA(
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
		float *knnmap_image_out);

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> OriginalRasterizeGaussiansConvertImageCUDA(
		const torch::Tensor &depth_buffer,
		const torch::Tensor &normal_buffer,
		const torch::Tensor &seg_buffer,
		const torch::Tensor &knnmap_buffer);

	void print_tensor_info(const torch::Tensor &tensor);

	void test(
		float *rgb_buffer,
		int H, int W,
		float *rgb_image_out);

}