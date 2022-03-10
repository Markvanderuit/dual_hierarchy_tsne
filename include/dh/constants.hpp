/*
 * MIT License
 *
 * Copyright (c) 2021 Mark van de Ruit (Delft University of Technology)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

/**
 * Major/minor version numbers.
 */
#define DH_VERSION_MAJOR 1
#define DH_VERSION_MINOR 0

/**
 * Formatted version number
 */
#define DH_STR_EXPAND(x) #x
#define DH_STR(x) DH_STR_EXPAND(x)
#define DH_VERSION DH_STR(DH_VERSION_MAJOR) "." DH_STR(DH_VERSION_MINOR)

/**
 * Enable the following checks, throwing an exception on failure
 * - Runtime checks for correct component use
 * - OpenGL shader compilation errors
 * - OpenGL error code checking
 * - CUDA error code checking
 */
#define DH_ENABLE_ASSERT

/**
 * Enable OpenGL debug output and throw an exception on failure. In addition:
 * - Also support warnings
 */
#define DH_ENABLE_DEBUG
#define DH_ENABLE_DEBUG_WARNINGS


/**
 * Settings for anything handled by dh::util::Logger
 */
#define DH_LOG_TIMESTAMPS       // Prepend UTC timestmaps to every log as prefix
#define DH_LOG_PREFIX_WIDTH 20  // Have a minimum padding for every log's prefix


/**
 * Settings for visualization library and demo
 */
#define DH_ENABLE_VIS_IMGUI                   // Include ImGui based gui components
#define DH_ENABLE_VIS_EMBEDDING               // Include embedding render task
#define DH_ENABLE_VIS_EMBEDDING_HIERARCHY     // Include embedding hierarchy render task
#define DH_ENABLE_VIS_FIELD_HIERARCHY         // Include field hierarchy render task
#define DH_VIS_EMBEDDING_INIT true            // Enable embedding render task by default
#define DH_VIS_EMBEDDING_HIERARCHY_INIT false // Enable embedding hierarchy render task by default
#define DH_VIS_FIELD_HIERARCHY_INIT false     // Enable field hierarchy render task by default

/**
 * Settings for dual-hierarchy approximation
 */
#define DH_HIER_INIT_LVL_2D 3       // Top-down level at which dual-hierarchy traversal starts, 2D
#define DH_HIER_INIT_LVL_3D 2       // Top-down level at which dual-hierarchy traversal starts, 3D
#define DH_HIER_LVL_DIFFERENCE 4    // Max hierarchy depth difference where DH approximation is used
#define DH_BVH_REFIT_ITERS 4        // Max nr. of refits for BVH before a full rebuild is performed
#define DH_BVH_REFIT_PADDING 150    // Nr. of iters after early exagg. where refit is not allowed
#define DH_WORKQUEUE_HIGHWATER 1024 // Graphics memory in mb that is unavailable for work queues