# AGC_final
### download weights
You can download the pre-trained weight files in this [link](https://drive.google.com/drive/folders/10pJw5Bx80zDLEdsDjav5P6bCAUr2kCKk?usp=sharing), and put it in the AGC_final directory. 

### environments
torch==1.8.1<br/>
opencv<br/>
albumentations<br/>
timm<br/>
python-bidi<br/>
soynlp<br/>
pandas<br/>
matplotlib<br/>
seaborn<br/>

### inference code
1. download the pre-trained .pth weight files in the save_models. There are many pre-trained models for running the code.
2. run the prediction code, and specify your own video dataset directory with pre-defined.json. We provide the sample video.mp4 and pre-defined.json files.
```bash
python main.py --video_root_dir sample_dataset
```
3. You can see the predicted json files in the ./debug/answer.
4. You can also see the first ocr prediction results in the ./debug/first_ocr_results


### first frame ocr results
![image](https://user-images.githubusercontent.com/55695350/207616454-d16d45b8-b528-479d-bcd4-c1d0fa8ad386.png)
![image](https://user-images.githubusercontent.com/55695350/207616486-c8144f59-c1ee-49bc-bb4b-5c655537c761.png)


### fallen person detection, classification and indicating where the perfon falls.
![image](https://user-images.githubusercontent.com/55695350/207606510-092fbc36-e9c0-44b7-b39a-5652ef92052c.png)


https://user-images.githubusercontent.com/55695350/207610421-147a320f-982e-4f99-a1e9-d00427122ec9.mp4

