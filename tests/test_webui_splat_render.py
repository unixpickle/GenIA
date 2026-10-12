"""Render the actual GLSL shaders and compare with an independent CPU projection."""
import numpy as np

from webui_gles import GLES
from test_webui_splats import gaussian_data, reference_texture


def test_splat_shaders_match_cpu_projection_and_compositing():
    size=64
    data=gaussian_data(5)
    # Asymmetric ellipses at overlapping depths, including an off-axis center.
    for n in ['x','y','z']: data[n] *= .25
    data['scale_0']=-2; data['scale_1']=-2.8; data['scale_2']=-3
    texture=reference_texture(data)
    angle=.6;c=np.cos(angle);s=np.sin(angle)
    view=np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,-4.8],[0,0,0,1]])
    renderer=GLES(size)
    try: actual=renderer.draw(texture,view)[:,:,:3]/255
    finally: renderer.close()
    focal=size/(2*np.tan(np.pi/8))
    camera=texture[:,:3]@view[:3,:3].T+view[:3,3]
    yy,xx=np.mgrid[:size,:size]
    screen=np.stack([xx+.5-size/2,size/2-(yy+.5)],-1)
    expected=np.ones((size,size,3))
    for i in np.argsort(camera[:,2]):
        x,y,z=camera[i];d=-z;t=texture[i]
        cov=np.array([[t[4],t[5],t[6]],[t[5],t[8],t[9]],[t[6],t[9],t[10]]])
        jac=np.array([[focal/d,0,focal*x/d**2],[0,focal/d,focal*y/d**2]])@view[:3,:3]
        cov2=jac@cov@jac.T+np.eye(2)*.3
        delta=screen-focal*camera[i,:2]/d
        exponent=np.einsum('...i,ij,...j->...',delta,np.linalg.inv(cov2),delta)
        alpha=np.minimum(.99,t[3]*np.exp(-.5*exponent))
        alpha[alpha<1/255]=0
        expected=t[12:15]*alpha[...,None]+expected*(1-alpha[...,None])
    # RGBA8 framebuffer rounds each blend; allow the accumulated quantization.
    np.testing.assert_allclose(actual,expected,atol=4/255,rtol=0)
    assert actual.min()<.7, 'Renderer produced an empty image'
