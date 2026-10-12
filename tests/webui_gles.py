"""Optional headless GLES check using Chrome's ANGLE/SwiftShader libraries.

No browser/server required. Set GENIA_EGL_LIBRARY_DIR to a directory containing
libEGL and libGLESv2, or use a macOS Chrome installation that bundles them.
"""
import ctypes as c
import os
import re
from pathlib import Path

import numpy as np
import pytest


class GLES:
    def __init__(self, size):
        configured = os.environ.get('GENIA_EGL_LIBRARY_DIR')
        roots = [Path(configured)] if configured else list(Path('/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Versions').glob('*/Libraries'))
        libraries = [(p/egl, p/gles) for p in roots for egl, gles in
                     [('libEGL.dylib', 'libGLESv2.dylib'), ('libEGL.so', 'libGLESv2.so')]
                     if (p/egl).exists() and (p/gles).exists()]
        if not libraries: pytest.skip('ANGLE GLES libraries not available')
        self.egl = c.CDLL(str(libraries[0][0])); self.gl = c.CDLL(str(libraries[0][1]))
        self.size = size
        self.egl.eglGetProcAddress.argtypes = [c.c_char_p]; self.egl.eglGetProcAddress.restype = c.c_void_p
        get_display = c.CFUNCTYPE(c.c_void_p, c.c_uint, c.c_void_p, c.POINTER(c.c_int))(self.egl.eglGetProcAddress(b'eglGetPlatformDisplayEXT'))
        # ANGLE Vulkan backend, SwiftShader device: requires no GPU permissions.
        self.display = get_display(0x3202, None, (c.c_int*5)(0x3203,0x3450,0x3209,0x3487,0x3038))
        self.egl.eglInitialize.argtypes = [c.c_void_p,c.c_void_p,c.c_void_p]
        if not self.egl.eglInitialize(self.display,None,None): pytest.skip('Headless EGL unavailable')
        self.egl.eglChooseConfig.argtypes = [c.c_void_p,c.POINTER(c.c_int),c.POINTER(c.c_void_p),c.c_int,c.POINTER(c.c_int)]
        config = c.c_void_p(); count = c.c_int()
        attrs=(c.c_int*13)(0x3024,8,0x3023,8,0x3022,8,0x3021,8,0x3033,1,0x3040,0x40,0x3038)
        assert self.egl.eglChooseConfig(self.display,attrs,c.byref(config),1,c.byref(count)) and count.value
        self.egl.eglCreatePbufferSurface.argtypes=[c.c_void_p,c.c_void_p,c.POINTER(c.c_int)]
        self.egl.eglCreatePbufferSurface.restype=c.c_void_p
        self.surface=self.egl.eglCreatePbufferSurface(self.display,config,(c.c_int*5)(0x3057,size,0x3056,size,0x3038))
        self.egl.eglCreateContext.argtypes=[c.c_void_p,c.c_void_p,c.c_void_p,c.POINTER(c.c_int)]
        self.egl.eglCreateContext.restype=c.c_void_p
        self.context=self.egl.eglCreateContext(self.display,config,None,(c.c_int*3)(0x3098,3,0x3038))
        self.egl.eglMakeCurrent.argtypes=[c.c_void_p]*4
        assert self.egl.eglMakeCurrent(self.display,self.surface,self.surface,self.context)
        self.program=None

    def fn(self, name, result, *args):
        fn = getattr(self.gl, name); fn.restype=result; fn.argtypes=list(args); return fn

    def draw(self, texture, view, *, size=None):
        size = size or self.size
        source=(Path(__file__).resolve().parents[1]/'core/webui/static/splat-viewer.js').read_text()
        program=self.fn('glCreateProgram',c.c_uint)(); self.program=program
        for name,kind in [('vertexShader',0x8B31),('fragmentShader',0x8B30)]:
            shader=self.fn('glCreateShader',c.c_uint,c.c_uint)(kind)
            code=re.search(r'const '+name+r' = `([\s\S]*?)`;',source)[1].encode()
            string=c.c_char_p(code)
            self.fn('glShaderSource',None,c.c_uint,c.c_int,c.POINTER(c.c_char_p),c.c_void_p)(shader,1,c.byref(string),None)
            self.fn('glCompileShader',None,c.c_uint)(shader)
            ok=c.c_int();self.fn('glGetShaderiv',None,c.c_uint,c.c_uint,c.POINTER(c.c_int))(shader,0x8B81,c.byref(ok))
            log=c.create_string_buffer(4096)
            self.fn('glGetShaderInfoLog',None,c.c_uint,c.c_int,c.c_void_p,c.c_char_p)(shader,4096,None,log)
            assert ok.value, log.value.decode()
            self.fn('glAttachShader',None,c.c_uint,c.c_uint)(program,shader)
            self.fn('glDeleteShader',None,c.c_uint)(shader)
        self.fn('glLinkProgram',None,c.c_uint)(program)
        ok=c.c_int();self.fn('glGetProgramiv',None,c.c_uint,c.c_uint,c.POINTER(c.c_int))(program,0x8B82,c.byref(ok));assert ok.value
        self.fn('glUseProgram',None,c.c_uint)(program)
        view=np.asarray(view,dtype=np.float32).reshape(4,4)
        centers=texture[:,:3]; depths=(centers@view[:3,:3].T)[:,2]
        order=np.argsort(depths,kind='stable').astype(np.uint32)
        for name,values,integer in [('corner',np.array([-1,-1,1,-1,-1,1,1,1],np.float32),False),('splatIndex',order,True)]:
            buffer=c.c_uint();self.fn('glGenBuffers',None,c.c_int,c.POINTER(c.c_uint))(1,c.byref(buffer))
            self.fn('glBindBuffer',None,c.c_uint,c.c_uint)(0x8892,buffer.value)
            self.fn('glBufferData',None,c.c_uint,c.c_ssize_t,c.c_void_p,c.c_uint)(0x8892,values.nbytes,values.ctypes.data,0x88E4)
            loc=self.fn('glGetAttribLocation',c.c_int,c.c_uint,c.c_char_p)(program,name.encode())
            self.fn('glEnableVertexAttribArray',None,c.c_uint)(loc)
            if integer:
                self.fn('glVertexAttribIPointer',None,c.c_uint,c.c_int,c.c_uint,c.c_int,c.c_void_p)(loc,1,0x1405,0,None)
                self.fn('glVertexAttribDivisor',None,c.c_uint,c.c_uint)(loc,1)
            else:
                self.fn('glVertexAttribPointer',None,c.c_uint,c.c_int,c.c_uint,c.c_ubyte,c.c_int,c.c_void_p)(loc,2,0x1406,False,0,None)
        width=2048;height=(len(texture)*4+width-1)//width
        pixels=np.zeros(width*height*4,np.float32);pixels[:texture.size]=texture.reshape(-1)
        tex=c.c_uint();self.fn('glGenTextures',None,c.c_int,c.POINTER(c.c_uint))(1,c.byref(tex))
        self.fn('glBindTexture',None,c.c_uint,c.c_uint)(0x0DE1,tex.value)
        for key in [0x2800,0x2801]:self.fn('glTexParameteri',None,c.c_uint,c.c_uint,c.c_int)(0x0DE1,key,0x2600)
        self.fn('glTexImage2D',None,c.c_uint,c.c_int,c.c_int,c.c_int,c.c_int,c.c_int,c.c_uint,c.c_uint,c.c_void_p)(0x0DE1,0,0x8814,width,height,0,0x1908,0x1406,pixels.ctypes.data)
        uniform=lambda name:self.fn('glGetUniformLocation',c.c_int,c.c_uint,c.c_char_p)(program,name.encode())
        self.fn('glUniform1i',None,c.c_int,c.c_int)(uniform('splats'),0)
        self.fn('glUniform1i',None,c.c_int,c.c_int)(uniform('textureWidth'),width)
        matrix=np.ascontiguousarray(view.T)
        self.fn('glUniformMatrix4fv',None,c.c_int,c.c_int,c.c_ubyte,c.c_void_p)(uniform('view'),1,False,matrix.ctypes.data)
        self.fn('glUniform2f',None,c.c_int,c.c_float,c.c_float)(uniform('viewport'),size,size)
        self.fn('glUniform1f',None,c.c_int,c.c_float)(uniform('focal'),size/(2*np.tan(np.pi/8)))
        self.fn('glViewport',None,c.c_int,c.c_int,c.c_int,c.c_int)(0,0,size,size)
        self.fn('glClearColor',None,*([c.c_float]*4))(1,1,1,1)
        self.fn('glClear',None,c.c_uint)(0x4000)
        self.fn('glEnable',None,c.c_uint)(0x0BE2)
        self.fn('glBlendFunc',None,c.c_uint,c.c_uint)(1,0x0303)
        self.fn('glDrawArraysInstanced',None,c.c_uint,c.c_int,c.c_int,c.c_int)(5,0,4,len(texture))
        image=np.empty((size,size,4),np.uint8)
        self.fn('glReadPixels',None,c.c_int,c.c_int,c.c_int,c.c_int,c.c_uint,c.c_uint,c.c_void_p)(0,0,size,size,0x1908,0x1401,image.ctypes.data)
        assert self.fn('glGetError',c.c_uint)()==0
        return image[::-1].copy()

    def close(self):
        self.egl.eglMakeCurrent(self.display,None,None,None)
        for name,handle in [('eglDestroyContext',self.context),('eglDestroySurface',self.surface)]:
            fn=getattr(self.egl,name);fn.argtypes=[c.c_void_p,c.c_void_p];fn(self.display,handle)
        self.egl.eglTerminate.argtypes=[c.c_void_p];self.egl.eglTerminate(self.display)
