# CB-detection
Source codes for detecting Cardiovascular Borderline(CB) in chest X-ray. (ver.1)
<br/>
<br/>
<br/>

## Environments
Ubuntu 18.04 LTS, Python 3.6.7, Keras 2.1.6, TensorFlow 1.9, CUDA 9.0, cuDNN 7.0 <br/>
Or you can use [*Dockerfile*](./Dockerfile).
<br/>
<br/>

## Trained weights
[Mask-RCNN](https://arxiv.org/pdf/1703.06870.pdf) was used to detect CB. Pre-trained weights with [MS COCO](https://ttic.uchicago.edu/~mmaire/papers/pdf/coco_eccv2014.pdf) were fine-tuned using chest x-ray dataset. <br/>
Our CB detection weights can be downloaded from [*Google Drive*](https://drive.google.com/drive/folders/1jHI1ftloX7qrztCvMu3g6YZTnEz-gjFM?usp=sharing) here.
<br/>
<br/>

## CB Detection
Codes for detecting CB with the trained weights are implemented in [*src/run.py*](./src/run.py).
<br/>
<p>
<img src="https://user-images.githubusercontent.com/17020746/125767728-68b90a14-ec84-42e7-9f5e-cb28802de6c3.png" width="40%">
</p>

When the above X-ray PNG([*data/1000137.png*](./data/1000137.png)) is input to the model, the following masks are output. <br/>
From left to right, Aortic Knob, Carina, DAO, LAA, Lt Lower CB, Pulmonary Conus, Rt Lower CB, Rt Upper CB.<br/>
<br/>

<p>
<img src="https://user-images.githubusercontent.com/17020746/125766576-8f016720-113d-44c3-96c9-d11f599a3a9f.png" width="12%">  
<img src="https://user-images.githubusercontent.com/17020746/125766667-3e2f89bc-5a97-4643-8ce5-7e8979c406c3.png" width="12%">  
<img src="https://user-images.githubusercontent.com/17020746/125766832-2708e17b-e9c3-4d91-a628-5c65372514f4.png" width="12%">  
<img src="https://user-images.githubusercontent.com/17020746/125766895-d72c5513-91d5-44f3-bd9b-5e25cfeda87d.png" width="12%">  
<img src="https://user-images.githubusercontent.com/17020746/125766961-27df336b-6010-4d3f-a805-68b2fe0c047f.png" width="12%">  
<img src="https://user-images.githubusercontent.com/17020746/125767003-94164ef5-0f56-4855-a683-a9d46c9cd71d.png" width="12%">  
<img src="https://user-images.githubusercontent.com/17020746/125767050-845fe82b-ee4f-48ce-85d6-bb64af61e693.png" width="12%">  
<img src="https://user-images.githubusercontent.com/17020746/125767236-d3d4df10-f654-4950-90a7-8eac88500633.png" width="12%">
</p>

<br/>

## DIY
The other images are in the [data](./data) directory, so you can do additional inference yourself by following the tutorial below.<br/>

#### [Step1. Setting env] : Build docker image & Run container
Just run [*run.sh*](./run.sh) flie in linux.<br/>
This creates a docker image(cbdetection:2.2.0) and runs a container (cont.cbdetection).
<br/>
#### [Step2. Run inference code]
Execute the python script([*run.py*](./src/run.py)) referring to the command example below.
<pre>python run.py \
--path_img='/data' \
--weight_path='./weights/model.seg.cb.hdf5' \
--path_dst='./results'<code>
<br/><br/>


