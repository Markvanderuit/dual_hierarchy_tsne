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

/**
 * GLSL(name, version, ...)
 * 
 * GLSL verbatism wrapper, embedding GLSL as a string literal in C++ code. If
 * preprocessor code is involved in GLSL, this must be wrapped in a GLSL_PROTECT()
 * statement.
 *  
 * @author M. Billeter
 * @param name - name of string that becomes embedded
 * @param version - required version of shader, eg. 450
 * @param ... - misc arguments contain shader code.
 */
#define GLSL(name, version, ...) \
  static const char * name = \
  "#version " #version "\n" GLSL_IMPL(__VA_ARGS__)
#define GLSL_IMPL(...) #__VA_ARGS__

/**
 * GLSL_PROTECT(...)
 * 
 * Helper wrapper to allow embedding of GLSL preprocessor statements
 * (eg. #extension) in embedded GLSL code.
 * 
 * @author M. Billeter
 * @param ... - statement to wrap
 */
#define GLSL_PROTECT(...) \n __VA_ARGS__ \n

// Allow expansion of hash symbols into preprocessor staements
#define HASH_SMB #
#define HASH_EXP(x) x

/**
 * GLSL_DEFINE(...), GLSL_IFDEF(...), GLSL_ENDIF()
 * 
 * Wrappers for embedding GLSL DEFINE/IFDEF/ENDIF statements in embedded
 * and/or stringified GLSL code without the C preprocessor intervening.
 * 
 * @param ... - preprocessor define/ifdef to warp
 */
#define GLSL_DEFINE(...) \n HASH_EXP(HASH_SMB) ## define __VA_ARGS__ \n
#define GLSL_IFDEF(...) \n HASH_EXP(HASH_SMB) ## ifdef __VA_ARGS__ \n
#define GLSL_ENDIF() \n HASH_EXP(HASH_SMB) ## endif \n

/**
 * GLSL_IF_DEF(...)
 * 
 * Wrapper for embedding #IFDEF CODE #ENDIF lines in embedded
 * and/or stringified GLSL code without the C preprocessor intervening.
 * 
 * @param name - preprocessor define to toggle wrapped code
 * @param ... - code to be wrapped
 */
#define GLSL_IF_DEF(name, ...)\
  GLSL_IFDEF(name)\
  GLSL_PROTECT(__VA_ARGS__)\
  GLSL_ENDIF()