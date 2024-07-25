# RTMPose-NCNN

# Introduction
**RTMPose** using ncnn easier ！no complicated steps required，**FPS 30+** running on Raspberry Pi 5<br>

![infer show](./results.gif)
# How to run
``` shell
mkdir build
cd build 
cmake ..
make 
./rtm-ncnn 1 ../000000000785_0.jpg
./rtm-ncnn 0 0 #use cam
```

## reference
<https://github.com/Tencent/ncnn><br>
