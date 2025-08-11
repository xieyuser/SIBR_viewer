/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#include <projects/gaussianviewer/renderer/ImageConvert.hpp>
#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>
#include <rasterizer.h>
#include <imgui_internal.h>

// Define the types and sizes that make up the contents of each Gaussian
// in the trained model.
typedef sibr::Vector3f Pos;

template <int D>
struct SHs
{
	float shs[(D + 1) * (D + 1) * 3];
};

template <int S>
struct Segs
{
	float seg[S];
};

struct Scale
{
	float scale[3];
};

struct Rot
{
	float rot[4];
};

template <int D, int S>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	Segs<S> segs;
	float opacity;
	float contri;
	Scale scale;
	Rot rot;
};

float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
	return log(m1 / (1.0f - m1));
}

// Load the Gaussians from the given file.
template <int D, int S>
int loadPly(
	const char *filename,
	std::vector<Pos> &pos,
	std::vector<SHs<3>> &shs,
	std::vector<Segs<15>> &segs,
	std::vector<float> &opacities,
	std::vector<float> &contri,
	std::vector<Scale> &scales,
	std::vector<Rot> &rot,
	sibr::Vector3f &minn,
	sibr::Vector3f &maxx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		SIBR_ERR << "Unable to find model's PLY file, attempted:\n"
				 << filename << std::endl;

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	// Output number of Gaussians contained
	SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<D, S>> points(count);
	infile.read((char *)points.data(), count * sizeof(RichPoint<D, S>));

	// Resize our SoA data
	pos.resize(count);
	shs.resize(count);
	segs.resize(count);
	scales.resize(count);
	rot.resize(count);
	opacities.resize(count);
	contri.resize(count);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile
	// (close in 3D --> close in 2D).
	minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		sibr::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		sibr::Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++)
		{
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair<uint64_t, int> &a, const std::pair<uint64_t, int> &b)
	{
		return a.first < b.first;
	};
	std::sort(mapp.begin(), mapp.end(), sorter);

	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);

	for (int k = 0; k < count; k++)
	{
		int i = mapp[k].second;
		pos[k] = points[i].pos;

		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			scales[k].scale[j] = exp(points[i].scale.scale[j]);

		for (int s_i = 0; s_i < 15; s_i++)
		{
			segs[k].seg[s_i] = points[i].segs.seg[s_i];
		}

		// Activate alpha
		opacities[k] = sigmoid(points[i].opacity);
		contri[k] = sigmoid(points[i].contri);
		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];

		for (int j = 1; j < SH_N; j++)
		{
			shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
		}
	}
	return count;
}

void savePly(const char *filename,
			 const std::vector<Pos> &pos,
			 const std::vector<SHs<3>> &shs,
			 const std::vector<float> &opacities,
			 const std::vector<Scale> &scales,
			 const std::vector<Rot> &rot,
			 const sibr::Vector3f &minn,
			 const sibr::Vector3f &maxx)
{
	// Read all Gaussians at once (AoS)
	int count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		count++;
	}
	std::vector<RichPoint<3, 15>> points(count);

	// Output number of Gaussians contained
	SIBR_LOG << "Saving " << count << " Gaussian splats" << std::endl;

	std::ofstream outfile(filename, std::ios_base::binary);

	outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";

	std::string props1[] = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
	std::string props2[] = {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};

	for (auto s : props1)
		outfile << "property float " << s << std::endl;
	for (int i = 0; i < 45; i++)
		outfile << "property float f_rest_" << i << std::endl;
	for (auto s : props2)
		outfile << "property float " << s << std::endl;
	outfile << "end_header" << std::endl;

	count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		points[count].pos = pos[i];
		points[count].rot = rot[i];
		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			points[count].scale.scale[j] = log(scales[i].scale[j]);
		// Activate alpha
		points[count].opacity = inverse_sigmoid(opacities[i]);
		points[count].shs.shs[0] = shs[i].shs[0];
		points[count].shs.shs[1] = shs[i].shs[1];
		points[count].shs.shs[2] = shs[i].shs[2];
		for (int j = 1; j < 16; j++)
		{
			points[count].shs.shs[(j - 1) + 3] = shs[i].shs[j * 3 + 0];
			points[count].shs.shs[(j - 1) + 18] = shs[i].shs[j * 3 + 1];
			points[count].shs.shs[(j - 1) + 33] = shs[i].shs[j * 3 + 2];
		}
		count++;
	}
	outfile.write((char *)points.data(), sizeof(RichPoint<3, 15>) * points.size());
}

namespace sibr
{
	// A simple copy renderer class. Much like the original, but this one
	// reads from a buffer instead of a texture and blits the result to
	// a render target.
	class BufferCopyRenderer
	{

	public:
		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
						 sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
						 sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		void process(uint bufferID, IRenderTarget &dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool &flip() { return _flip.get(); }
		int &width() { return _width.get(); }
		int &height() { return _height.get(); }

	private:
		GLShader _shader;
		GLuniform<bool> _flip = false; ///< Flip the texture when copying.
		GLuniform<int> _width = 1000;
		GLuniform<int> _height = 800;
	};
}

std::function<char *(size_t N)> resizeFunctional(void **ptr, size_t &S)
{
	auto lambda = [ptr, &S](size_t N)
	{
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char *>(*ptr);
	};
	return lambda;
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr &ibrScene, uint render_w, uint render_h, const char *file, bool *messageRead, int sh_degree, bool white_bg, bool useInterop, int device) : _scene(ibrScene),
																																																		   _dontshow(messageRead),
																																																		   _sh_degree(sh_degree),
																																																		   sibr::ViewBase(render_w, render_h)
{
	int num_devices;

	// key modification
	useInterop = false;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
	_device = device;
	if (device >= num_devices)
	{
		if (num_devices == 0)
			SIBR_ERR << "No CUDA devices detected!";
		else
			SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
	}
	CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
	if (prop.major < 7)
	{
		SIBR_ERR << "Sorry, need at least compute capability 7.0+!";
	}

	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = render_w;
	_copyRenderer->height() = render_h;

	std::vector<uint> imgs_ulr;
	const auto &cams = ibrScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid)
	{
		if (cams[cid]->isActive())
		{
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

	// Load the PLY data (AoS) to the GPU (SoA)
	std::vector<Pos> pos;
	std::vector<Rot> rot;
	std::vector<Scale> scale;
	std::vector<float> opacity;
	std::vector<SHs<3>> shs;
	std::vector<float> contri;
	std::vector<Segs<15>> segs;
	if (sh_degree == 0)
	{
		count = loadPly<0, 15>(file, pos, shs, segs, opacity, contri, scale, rot, _scenemin, _scenemax);
	}
	else if (sh_degree == 1)
	{
		count = loadPly<1, 15>(file, pos, shs, segs, opacity, contri, scale, rot, _scenemin, _scenemax);
	}
	else if (sh_degree == 2)
	{
		count = loadPly<2, 15>(file, pos, shs, segs, opacity, contri, scale, rot, _scenemin, _scenemax);
	}
	else if (sh_degree == 3)
	{
		count = loadPly<3, 15>(file, pos, shs, segs, opacity, contri, scale, rot, _scenemin, _scenemax);
	}

	_boxmin = _scenemin;
	_boxmax = _scenemax;

	int P = count;

	// Allocate and fill the GPU data
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&pos_cuda, sizeof(Pos) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&rot_cuda, sizeof(Rot) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&shs_cuda, sizeof(SHs<3>) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&seg_cuda, sizeof(Segs<15>) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(seg_cuda, segs.data(), sizeof(Segs<15>) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&opacity_cuda, sizeof(float) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&contri_cuda, sizeof(float) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(contri_cuda, contri.data(), sizeof(float) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&scale_cuda, sizeof(Scale) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));

	// Create space for view parameters
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&rect_cuda, 2 * P * sizeof(int)));

	float bg[3] = {white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f};
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	gData = new GaussianData(
		P,
		(float *)pos.data(),
		(float *)rot.data(),
		(float *)scale.data(),
		opacity.data(),
		contri.data(),
		(float *)shs.data(),
		(float *)segs.data());

	_gaussianRenderer = new GaussianSurfaceRenderer();

	// Create GL buffer ready for CUDA/GL interop
	glCreateBuffers(1, &imageBuffer);
	glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	glCreateBuffers(1, &depthOutBuffer);
	glNamedBufferStorage(depthOutBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	glCreateBuffers(1, &normalOutBuffer);
	glNamedBufferStorage(normalOutBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	glCreateBuffers(1, &segOutBuffer);
	glNamedBufferStorage(segOutBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	glCreateBuffers(1, &alphaOutBuffer);
	glNamedBufferStorage(alphaOutBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	glCreateBuffers(1, &knnmapOutBuffer);
	glNamedBufferStorage(knnmapOutBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	fallback_rgb_bytes.resize(render_w * render_h * 3 * sizeof(float));
	cudaMalloc(&fallbackRGBBufferCuda, fallback_rgb_bytes.size());

	fallback_depth_bytes.resize(render_w * render_h * sizeof(float));
	cudaMalloc(&fallbackDepthBufferCuda, fallback_depth_bytes.size());

	fallback_normal_bytes.resize(render_w * render_h * 3 * sizeof(float));
	cudaMalloc(&fallbackNormalBufferCuda, fallback_normal_bytes.size());

	fallback_seg_bytes.resize(render_w * render_h * 15 * sizeof(float));
	cudaMalloc(&fallbackSegBufferCuda, fallback_seg_bytes.size());

	fallback_alpha_bytes.resize(render_w * render_h * sizeof(float));
	cudaMalloc(&fallbackAlphaBufferCuda, fallback_alpha_bytes.size());

	fallback_knnmap_bytes.resize(render_w * render_h * sizeof(float));
	cudaMalloc(&fallbackKnnmapBufferCuda, fallback_knnmap_bytes.size());

	fallback_depthout_bytes.resize(render_w * render_h * 3 * sizeof(float));
	cudaMalloc(&fallbackDepthOutBufferCuda, fallback_depthout_bytes.size());

	fallback_normalout_bytes.resize(render_w * render_h * 3 * sizeof(float));
	cudaMalloc(&fallbackNormalOutBufferCuda, fallback_normalout_bytes.size());

	fallback_segout_bytes.resize(render_w * render_h * 3 * sizeof(float));
	cudaMalloc(&fallbackSegOutBufferCuda, fallback_segout_bytes.size());

	fallback_alphaout_bytes.resize(render_w * render_h * 3 * sizeof(float));
	cudaMalloc(&fallbackAlphaOutBufferCuda, fallback_alphaout_bytes.size());

	fallback_knnmapout_bytes.resize(render_w * render_h * 3 * sizeof(float));
	cudaMalloc(&fallbackKnnmapOutBufferCuda, fallback_knnmapout_bytes.size());

	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);
}

void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr &newScene)
{
	_scene = newScene;

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto &cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid)
	{
		if (cams[cid]->isActive())
		{
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget &dst, const sibr::Camera &eye)
{
	if (currMode == "Ellipsoids")
	{
		_gaussianRenderer->process(count, *gData, eye, dst, 0.2f);
	}
	else if (currMode == "Initial Points")
	{
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	}
	else
	{
		// Convert view and projection to target coordinate system
		auto view_mat = eye.view();
		auto proj_mat = eye.viewproj();
		view_mat.row(1) *= -1;
		view_mat.row(2) *= -1;
		proj_mat.row(1) *= -1;

		// Compute additional view parameters
		float tan_fovy = tan(eye.fovy() * 0.5f);
		float tan_fovx = tan_fovy * eye.aspect();

		// Copy frame-dependent data to GPU
		CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));

		float *image_cuda = nullptr;
		float *segout_cuda = nullptr;
		float *depth_cuda = nullptr;
		float *normal_cuda = nullptr;
		float *alpha_cuda = nullptr;
		float *knnmap_cuda = nullptr;

		float *segoutout_cuda = nullptr;
		float *depthout_cuda = nullptr;
		float *normalout_cuda = nullptr;
		float *alphaout_cuda = nullptr;
		float *knnmapout_cuda = nullptr;

		// allocate memory
		image_cuda = fallbackRGBBufferCuda;
		depth_cuda = fallbackDepthBufferCuda;
		normal_cuda = fallbackNormalBufferCuda;
		segout_cuda = fallbackSegBufferCuda;
		alpha_cuda = fallbackAlphaBufferCuda;
		knnmap_cuda = fallbackKnnmapBufferCuda;

		depthout_cuda = fallbackDepthOutBufferCuda;
		normalout_cuda = fallbackNormalOutBufferCuda;
		segoutout_cuda = fallbackSegOutBufferCuda;
		alphaout_cuda = fallbackAlphaOutBufferCuda;
		knnmapout_cuda = fallbackKnnmapOutBufferCuda;

		// Rasterize
		// int *rects = _fastCulling ? rect_cuda : nullptr;
		// float *boxmin = _cropping ? (float *)&_boxmin : nullptr;
		// float *boxmax = _cropping ? (float *)&_boxmax : nullptr;

		// CudaRasterizer::Rasterizer::forward(
		// 	geomBufferFunc,
		// 	binningBufferFunc,
		// 	imgBufferFunc,
		// 	count, _sh_degree, 16,
		// 	background_cuda,
		// 	_resolution.x(),
		// 	_resolution.y(),
		// 	pos_cuda,
		// 	shs_cuda,
		// 	nullptr,
		// 	opacity_cuda,
		// 	scale_cuda,
		// 	_scalingModifier,
		// 	rot_cuda,
		// 	nullptr,
		// 	view_cuda,
		// 	proj_cuda,
		// 	cam_pos_cuda,
		// 	tan_fovx,
		// 	tan_fovy,
		// 	false,
		// 	image_cuda,
		// 	_antialiasing,
		// 	nullptr,
		// 	rects,
		// 	boxmin,
		// 	boxmax);

		CudaRasterizer::Rasterizer::forward(
			geomBufferFunc,
			binningBufferFunc,
			imgBufferFunc,
			count,
			_sh_degree,
			16,
			background_cuda,
			_resolution.x(),
			_resolution.y(),
			pos_cuda,
			shs_cuda,
			seg_cuda,
			nullptr,
			opacity_cuda,
			contri_cuda,
			scale_cuda,
			_scalingModifier,
			0.1f,
			rot_cuda,
			nullptr,
			view_cuda,
			proj_cuda,
			cam_pos_cuda,
			tan_fovx,
			tan_fovy,
			false,
			image_cuda,
			segout_cuda,
			depth_cuda,
			normal_cuda,
			alpha_cuda,
			knnmap_cuda,
			nullptr,
			false);

		RasterizeGaussiansConvertImageCUDA(
			// input
			depth_cuda,
			normal_cuda, 3,
			segout_cuda, 15,
			knnmap_cuda,
			_copyRenderer->height(), _copyRenderer->width(),
			// output
			depthout_cuda,
			normalout_cuda,
			segoutout_cuda,
			knnmapout_cuda);

		if (currModal == "depth")
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_depthout_bytes.data(), fallbackDepthOutBufferCuda, fallback_depthout_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(depthOutBuffer, 0, fallback_depthout_bytes.size(), fallback_depthout_bytes.data());
			_copyRenderer->process(depthOutBuffer, dst, _resolution.x(), _resolution.y());
		}
		else if (currModal == "normal")
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_normalout_bytes.data(), fallbackNormalOutBufferCuda, fallback_normalout_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(normalOutBuffer, 0, fallback_normalout_bytes.size(), fallback_normalout_bytes.data());
			_copyRenderer->process(normalOutBuffer, dst, _resolution.x(), _resolution.y());
		}
		else if (currModal == "semantic")
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_segout_bytes.data(), fallbackSegOutBufferCuda, fallback_segout_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(segOutBuffer, 0, fallback_segout_bytes.size(), fallback_segout_bytes.data());
			_copyRenderer->process(segOutBuffer, dst, _resolution.x(), _resolution.y());
		}
		else if (currModal == "knnmap")
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_knnmapout_bytes.data(), fallbackKnnmapOutBufferCuda, fallback_knnmapout_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(knnmapOutBuffer, 0, fallback_knnmapout_bytes.size(), fallback_knnmapout_bytes.data());
			_copyRenderer->process(knnmapOutBuffer, dst, _resolution.x(), _resolution.y());
		}
		else if (currModal == "alpha")
		{
			_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
		}
		else
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_rgb_bytes.data(), fallbackRGBBufferCuda, fallback_rgb_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(imageBuffer, 0, fallback_rgb_bytes.size(), fallback_rgb_bytes.data());
			_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
		}
	}

	if (cudaPeekAtLastError() != cudaSuccess)
	{
		SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
	}
}

void sibr::GaussianView::onUpdate(Input &input)
{
}

void sibr::GaussianView::onGUI()
{
	// Generate and update UI elements
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str()))
	{
		if (ImGui::BeginCombo("Render Mode", currMode.c_str()))
		{
			if (ImGui::Selectable("Splats"))
				currMode = "Splats";
			if (ImGui::Selectable("Initial Points"))
				currMode = "Initial Points";
			if (ImGui::Selectable("Ellipsoids"))
				currMode = "Ellipsoids";
			ImGui::EndCombo();
		}
		if (ImGui::BeginCombo("Modal", currModal.c_str()))
		{
			if (ImGui::Selectable("rgb"))
				currModal = "rgb";
			if (ImGui::Selectable("depth"))
				currModal = "depth";
			if (ImGui::Selectable("normal"))
				currModal = "normal";
			if (ImGui::Selectable("alpha"))
				currModal = "alpha";
			if (ImGui::Selectable("semantic"))
				currModal = "semantic";
			if (ImGui::Selectable("knnmap"))
				currModal = "knnmap";
			ImGui::EndCombo();
		}
	}
	if (currMode == "Splats")
	{
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);
	}
	ImGui::Checkbox("Fast culling", &_fastCulling);
	ImGui::Checkbox("Antialiasing", &_antialiasing);

	ImGui::Checkbox("Crop Box", &_cropping);
	if (_cropping)
	{
		ImGui::SliderFloat("Box Min X", &_boxmin.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Min Y", &_boxmin.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Min Z", &_boxmin.z(), _scenemin.z(), _scenemax.z());
		ImGui::SliderFloat("Box Max X", &_boxmax.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Max Y", &_boxmax.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Max Z", &_boxmax.z(), _scenemin.z(), _scenemax.z());
		ImGui::InputText("File", _buff, 512);
		if (ImGui::Button("Save"))
		{
			// std::vector<Pos> pos(count);
			// std::vector<Rot> rot(count);
			// std::vector<float> opacity(count);
			// std::vector<SHs<3>> shs(count);
			// std::vector<Scale> scale(count);
			// std::vector<Segs<15>> segs(count);
			// std::vector<float> contri(count);
			// CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos.data(), pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
			// CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot.data(), rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));
			// CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity.data(), opacity_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
			// CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(contri.data(), contri_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
			// CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs.data(), shs_cuda, sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
			// CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(segs.data(), seg_cuda, sizeof(Segs<15>) * count, cudaMemcpyDeviceToHost));
			// CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale.data(), scale_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));
			// savePly(_buff, pos, shs, opacity, scale, rot, _boxmin, _boxmax);
		}
	}

	ImGui::End();

	if (!*_dontshow && !accepted && _interop_failed)
		ImGui::OpenPopup("Error Using Interop");

	if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::SetItemDefaultFocus();
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"
					" It did NOT work for your current configuration.\n"
					" For highest performance, OpenGL and CUDA must run on the same\n"
					" GPU on an OS that supports interop.You can try to pass a\n"
					" non-zero index via --device on a multi-GPU system, and/or try\n"
					" attaching the monitors to the main CUDA card.\n"
					" On a laptop with one integrated and one dedicated GPU, you can try\n"
					" to set the preferred GPU via your operating system.\n\n"
					" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

		ImGui::Separator();

		if (ImGui::Button("  OK  "))
		{
			ImGui::CloseCurrentPopup();
			accepted = true;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Don't show this message again", _dontshow);
		ImGui::EndPopup();
	}
}

sibr::GaussianView::~GaussianView()
{
	// Cleanup
	cudaFree(pos_cuda);
	cudaFree(rot_cuda);
	cudaFree(scale_cuda);
	cudaFree(opacity_cuda);
	cudaFree(contri_cuda);
	cudaFree(shs_cuda);
	cudaFree(seg_cuda);

	cudaFree(view_cuda);
	cudaFree(proj_cuda);
	cudaFree(cam_pos_cuda);
	cudaFree(background_cuda);
	cudaFree(rect_cuda);

	cudaFree(fallbackRGBBufferCuda);
	cudaFree(fallbackDepthBufferCuda);
	cudaFree(fallbackNormalBufferCuda);
	cudaFree(fallbackSegBufferCuda);
	cudaFree(fallbackAlphaBufferCuda);
	cudaFree(fallbackKnnmapBufferCuda);

	cudaFree(fallbackDepthOutBufferCuda);
	cudaFree(fallbackNormalOutBufferCuda);
	cudaFree(fallbackSegOutBufferCuda);
	cudaFree(fallbackAlphaOutBufferCuda);
	cudaFree(fallbackKnnmapOutBufferCuda);

	glDeleteBuffers(1, &imageBuffer);
	glDeleteBuffers(1, &depthOutBuffer);
	glDeleteBuffers(1, &segOutBuffer);
	glDeleteBuffers(1, &normalOutBuffer);
	glDeleteBuffers(1, &alphaOutBuffer);
	glDeleteBuffers(1, &knnmapOutBuffer);

	if (geomPtr)
		cudaFree(geomPtr);
	if (binningPtr)
		cudaFree(binningPtr);
	if (imgPtr)
		cudaFree(imgPtr);

	delete _copyRenderer;
}
