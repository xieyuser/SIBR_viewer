# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr

set(SIBR_PROJECT "gaussian")
project(sibr_${SIBR_PROJECT} LANGUAGES CXX)

set(CMAKE_CUDA_ARCHITECTURES "80;86;90")

sibr_gitlibrary(TARGET CudaRasterizer
    GIT_REPOSITORY 	"https://github.com/xieyuser/gaussian-semantic-normal-rasterization-.git"
    GIT_TAG			"34220f01514317dad36c4fd4083f4a807908651f"
)

# sibr_gitlibrary(TARGET CudaRasterizer
#     GIT_REPOSITORY 	"https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
#     GIT_TAG			"255c532d5fd0c3d83afec0e8e49574b65ae381bd"
# )


find_package(CUDAToolkit REQUIRED)

include("${PROJECT_SOURCE_DIR}/cmake/torch.cmake")

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH}
$ENV{CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)


file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

file(GLOB SHADERS "shaders/*.frag" "shaders/*.vert" "shaders/*.geom")
source_group("Source Files\\shaders" FILES ${SHADERS})

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "shaders/*.frag" "shaders/*.vert" "shaders/*.geom")

## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES})

include_directories(${Boost_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} .)

message("dddddddddddddddddddddddddddddddddddddddddd"${TORCH_LIBRARIES})

if (WIN32)
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	${TORCH_LIBRARIES}
	glfw3
	sibr_system
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
	CUDA::cudart
	CudaRasterizer
)
else()
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	glfw
	sibr_system
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
	CUDA::cudart
	CudaRasterizer
	torch_cpu
    c10
)
endif()

add_definitions( -DSIBR_EXP_ULR_EXPORTS -DBOOST_ALL_DYN_LINK  )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "projects/${SIBR_PROJECT}/renderer")

## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
	SHADERS ${SHADERS}
	RSC_FOLDER ${SIBR_PROJECT}

    #STANDALONE  ${INSTALL_STANDALONE}   ## mean call install_runtime with bundle dependencies resolution
    COMPONENT   ${PROJECT_NAME}_install ## will create custom target to install only this project
)
