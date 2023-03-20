import torch, numpy
from torch.autograd import grad

def pp_identity(sample,*args):
    return sample

def pp_linearcombination(*args, **kwargs):
    """ possible parameters:
    sample:     the sample at time index i 
    origin:     the image with noise
    clear:      the original image without noise
    i:          the current time index
    T:          the totoal time index
    use_clear:  whether use the clear image as the reference image
    nonlinear:  the nonlinear function used to transfer the i/T
    """
    assert (use_clear and clear is not None) or (not use_clear)

    alpha = (i/T)
    alpha = nonlinear(alpha)

    reference = clear if use_clear else origin
    
    return alpha * origin + (1 - alpha) * reference

def pp_convexcombination(*args, **kwargs):
    """ possible parameters:
    sample:     the sample at time index i 
    origin:     the image with noise
    alpha:      convex comb coefficient, before the sample image, 这个在一定程度上是 保留 小 t或者是 小 t的一种扩散
    """

    sample = kwargs["sample"]
    origin = kwargs["origin"]
    alpha = kwargs["alpha"]
    
    return alpha * sample + (1 - alpha) * origin

def flatten(y):
    return y.reshape(y.shape[0], -1).T

class Hardness:
    def __init__(self, y, f = flatten):
        fy = f(y) # 需要保证完成之后 y 每一列是一个样本 
        self.mu = fy.mean(dim = 1, keepdim=True)
        self.cov = fy.cov()
        self.mu.requires_grad = False
        self.cov.requires_grad = False
        self.f = f

    def to(self,device):
        self.mu = self.mu.to(device)
        self.cov = self.cov.to(device)
    
    def cal(self, x):
        with torch.set_grad_enabled(True):
            x.requires_grad = True
            fx = self.f(x)
            batch_size = fx.shape[1]
            fyexp = self.mu.expand(-1, batch_size)
            diff = fx - fyexp
            loss = ((self.cov @ diff) * diff).sum() / 2 / fx.shape[0]
            g = grad(loss, x)[0]
        x.requires_grad = False
        return g.detach()

def pp_hardness(*args, **kwargs):
    """ possible parameters:
    sample:     the sample at time index i 
    hardness:   one instance of the hardness socre
    alpha:      convex comb coefficient, before the sample image, 这个在一定程度上是 保留 小 t或者是 小 t的一种扩散
    """
    sample = kwargs["sample"]
    hardness = kwargs["hardness"]
    alpha = kwargs["alpha"]
    
    referece = hardness.cal(sample)
    
    return alpha * sample + (1 - alpha) * referece # ATT 这里可能需要加一个系数 因为 梯度和 sample 的取值范围不匹配

def Fourier(x):
    x = x.detach().cpu().numpy()

    f = numpy.fft.rfft2(x)
    fshift = numpy.fft.fftshift(f)
    # res = numpy.log(numpy.abs(fshift))
    return fshift

def iFourier(x, clip = (-10,10)):
    ishift = numpy.fft.ifftshift(x)
    iimg = numpy.fft.irfft2(ishift)
    # iimg = numpy.abs(iimg)
    return torch.from_numpy(iimg).float().clamp(*clip)

class toZO:
    def __init__(self,x):
        size = x.shape[-1]
        mi = x.amin(dim = (-1,-2), keepdim=True)
        ma = x.amax(dim = (-1,-2), keepdim=True)
        self.mi = mi.expand(-1,-1,size,size)
        self.ma = ma.expand(-1,-1,size,size)

    def forward(self, x):
        return (x-self.mi)/(self.ma - self.mi)

    def backward(self,x):
        return self.mi + (self.ma - self.mi) * x

def pp_frequencydomaincombination(*args, **kwargs):
    """ possible parameters:
    sample:     the sample at time index i 
    origin:     the image with noise
    range:      the range preserved in the sample
    transform:  the transform function used, from image to frequency
    inverse_transform:  the inverse transform used, from frequency to image

    One potential problem here is that the frequency domain transform cannot be done by torch, so this can break the gradient chain.
    So if you would like to conduct a gardient-based optimzation, please bypass this processing.
    """
    sample = kwargs["sample"]
    origin = kwargs["origin"]
    range = kwargs["range"]
    transform = kwargs["transform"]
    inverse_transform = kwargs["inverse_transform"]

    # tozo = toZO(sample)

    # import pdb 
    # pdb.set_trace()

    # sample = tozo.forward(sample)
    # origin = tozo.forward(origin)

    device = sample.device

    sample_freq = transform(sample)
    referece_freq = transform(origin)

    size = sample_freq.shape[-2:]
    up = int(size[0] * range[1]), int(size[1] * range[1])
    down = int(size[0] * range[0]),  int(size[1] * range[0])

    mid = int(size[0] / 2),int(size[1] / 2)

    output_freq = referece_freq
    output_freq[:,:,mid[0] - up[0]:mid[0] + up[0],mid[1] - up[1]:mid[1] + up[1]] = sample_freq[:,:,mid[0] - up[0]:mid[0] + up[0],mid[1] - up[1]:mid[1] + up[1]]
    output_freq[:,:,mid[0] - down[0]:mid[0] + down[0],mid[1] - down[1]:mid[1] + down[1]] = referece_freq[:,:,mid[0] - down[0]:mid[0] + down[0],mid[1] - down[1]:mid[1] + down[1]]

    output = inverse_transform(output_freq)

    output = output.to(device)

    # output = tozo.backward(output)
    
    return output


def pp_lowpass(*args, **kwargs):
    """ possible parameters:
    sample:     the sample at time index i 
    origin:     the image with proper noise
    clear:      the original image without noise
    i:          the current time index
    T:          the totoal time index
    use_clear:  whether use the clear image as the reference image
    mode:       the mode used: smooth --> area+bicubic, sharp --> nearest+nearest
    scale:      the downsample sacle
    """

    assert (kwargs["use_clear"] and kwargs["clear"] is not None) or (not kwargs["use_clear"])

    reference = kwargs["clear"] if kwargs["use_clear"] else kwargs["origin"]

    if "scale" not in kwargs.keys() or kwargs["scale"] is None:
        scale = 4
    else:
        scale = kwargs["scale"]

    if kwargs["mode"] == "smooth":
        dmode = "area"
        umode = "bicubic"
    elif kwargs["mode"] == "sharp":
        dmode = "nearest"
        umode = "nearest"

    sample = kwargs["sample"]

    dsize = int(sample.shape[-1] / scale)
        
    reference_d = torch.nn.functional.interpolate(
            reference, 
            size = dsize, 
            mode=dmode)
    reference_u = torch.nn.functional.interpolate(
            reference_d, 
            size = sample.shape[-1], 
            mode=umode)
    
        
    sample_d = torch.nn.functional.interpolate(
            sample, 
            size = dsize, 
            mode=dmode)
    sample_u = torch.nn.functional.interpolate(
            sample_d, 
            size = sample.shape[-1], 
            mode=umode)
    sample_highfreq = sample - sample_u
    
    return (reference_u + sample_highfreq) # ATT sample 和 clear 都没有被锁定在一个范围内，差值均值大概为零，上下采样没有对取值分布有太大的影响，所以暂时不控制了


def pp_highpass(*args, **kwargs):
    """ possible parameters:
    sample:     the sample at time index i 
    origin:     the image with proper noise
    clear:      the original image without noise
    i:          the current time index
    T:          the totoal time index
    use_clear:  whether use the clear image as the reference image
    mode:       the mode used: smooth --> area+bicubic, sharp --> nearest+nearest
    scale:      the downsample sacle
    """

    assert (kwargs["use_clear"] and kwargs["clear"] is not None) or (not kwargs["use_clear"])

    reference = kwargs["clear"] if kwargs["use_clear"] else kwargs["origin"]

    if "scale" not in kwargs.keys() or kwargs["scale"] is None:
        scale = 4
    else:
        scale = kwargs["scale"]

    if kwargs["mode"] == "smooth":
        dmode = "area"
        umode = "bicubic"
    elif kwargs["mode"] == "sharp":
        dmode = "nearest"
        umode = "nearest"

    sample = kwargs["sample"]

    dsize = int(sample.shape[-1] / scale)
        
    reference_d = torch.nn.functional.interpolate(
            reference, 
            size = dsize, 
            mode=dmode)
    reference_u = torch.nn.functional.interpolate(
            reference_d, 
            size = sample.shape[-1], 
            mode=umode)
    reference_highfreq = reference - reference_u
    
        
    sample_d = torch.nn.functional.interpolate(
            sample, 
            size = dsize, 
            mode=dmode)
    sample_u = torch.nn.functional.interpolate(
            sample_d, 
            size = sample.shape[-1], 
            mode=umode)
    
    return (sample_u + reference_highfreq)

def pp_midpass(*args, **kwargs):
    """ possible parameters:
    sample:     the sample at time index i 
    origin:     the image with proper noise
    clear:      the original image without noise
    i:          the current time index
    T:          the totoal time index
    use_clear:  whether use the clear image as the reference image
    mode:       the mode used: smooth --> area+bicubic, sharp --> nearest+nearest
    scale:      the downsample sacle
    """

    assert (kwargs["use_clear"] and kwargs["clear"] is not None) or (not kwargs["use_clear"])

    reference = kwargs["clear"] if kwargs["use_clear"] else kwargs["origin"]


    # use the reference mid, generate the low and high

    scale_l=16
    scale_h=4

    if kwargs["mode"] == "smooth":
        dmode = "area"
        umode = "bicubic"
    elif kwargs["mode"] == "sharp":
        dmode = "nearest"
        umode = "nearest"

    sample = kwargs["sample"]

    dsize_l = int(sample.shape[-1] / scale_l)
    dsize_h = int(sample.shape[-1] / scale_h)
        
    reference_d_l = torch.nn.functional.interpolate(
            reference, 
            size = dsize_l, 
            mode=dmode)
    reference_u_l = torch.nn.functional.interpolate(
            reference_d_l, 
            size = sample.shape[-1], 
            mode=umode)

    reference_d_h = torch.nn.functional.interpolate(
            reference, 
            size = dsize_h, 
            mode=dmode)
    reference_u_h = torch.nn.functional.interpolate(
            reference_d_h, 
            size = sample.shape[-1], 
            mode=umode)

    reference_midfreq = reference_u_h - reference_u_l
    
    sample_d_l = torch.nn.functional.interpolate(
            sample, 
            size = dsize_l, 
            mode=dmode)
    sample_u_l = torch.nn.functional.interpolate(
            sample_d_l, 
            size = sample.shape[-1], 
            mode=umode)
        
    sample_d_h = torch.nn.functional.interpolate(
            sample, 
            size = dsize_h, 
            mode=dmode)
    sample_u_h = torch.nn.functional.interpolate(
            sample_d_h, 
            size = sample.shape[-1], 
            mode=umode)

    sample_highfreq = sample - sample_u_h
    
    return (sample_u_l + reference_midfreq + sample_highfreq)