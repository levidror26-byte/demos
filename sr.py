import subprocess, torch, torch.nn as nn, torch.nn.functional as F, os, time, sys
os.makedirs('/content/w', exist_ok=True)
for n in ['realesr-general-x4v3.pth', 'realesr-general-wdn-x4v3.pth']:
    if not os.path.exists('/content/w/'+n):
        subprocess.run(['wget','-q','-O','/content/w/'+n,'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/'+n], check=True)
CL = sys.argv[1].split(',') if len(sys.argv) > 1 else ['c1','n2','n3','n4']
for c in CL:
    if not os.path.exists(f'/content/{c}.mp4'):
        subprocess.run(['wget','-q','-O',f'/content/{c}.mp4',f'https://raw.githubusercontent.com/levidror26-byte/demos/src/{c}.mp4'], check=True)
class SRV(nn.Module):
    def __init__(s, nf=64, nc=32, up=4):
        super().__init__(); s.up = up
        b = [nn.Conv2d(3,nf,3,1,1), nn.PReLU(nf)]
        for _ in range(nc): b += [nn.Conv2d(nf,nf,3,1,1), nn.PReLU(nf)]
        b += [nn.Conv2d(nf,3*up*up,3,1,1)]
        s.body = nn.ModuleList(b); s.ps = nn.PixelShuffle(up)
    def forward(s, x):
        o = x
        for m in s.body: o = m(o)
        return s.ps(o) + F.interpolate(x, scale_factor=s.up, mode='nearest')
def ld(p):
    d = torch.load(p, map_location='cpu', weights_only=False)
    return d.get('params', d)
a = ld('/content/w/realesr-general-x4v3.pth'); b = ld('/content/w/realesr-general-wdn-x4v3.pth')
DN = float(os.environ.get('DN', '0.5'))
net = SRV(); net.load_state_dict({k: DN*a[k] + (1-DN)*b[k] for k in a}); net = net.half().cuda().eval()
W, H, OW, OH = 1280, 720, 2560, 1440
for c in CL:
    t = time.time()
    dec = subprocess.Popen(['ffmpeg','-loglevel','error','-i',f'/content/{c}.mp4','-vf',f'scale={W}:{H}:flags=area','-f','rawvideo','-pix_fmt','rgb24','-'], stdout=subprocess.PIPE)
    enc = subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}','-r','24','-i','-',
                            '-c:v','libx264','-preset','medium','-crf','12','-pix_fmt','yuv420p',f'/content/{c}_sr.mp4'], stdin=subprocess.PIPE)
    n = 0
    with torch.no_grad():
        while True:
            buf = dec.stdout.read(W*H*3)
            if len(buf) < W*H*3: break
            x = torch.frombuffer(bytearray(buf), dtype=torch.uint8).cuda().view(H, W, 3).permute(2, 0, 1)[None].half() / 255
            y = net(x).clamp(0, 1).float()
            y = F.interpolate(y, size=(OH, OW), mode='area')
            enc.stdin.write((y[0].permute(1, 2, 0) * 255 + 0.5).byte().cpu().numpy().tobytes()); n += 1
    enc.stdin.close(); enc.wait(); dec.wait()
    print(c, n, 'frames', round(time.time()-t, 1), 's', round(os.path.getsize(f'/content/{c}_sr.mp4')/1e6, 1), 'MB', flush=True)
print('ALL DONE')
