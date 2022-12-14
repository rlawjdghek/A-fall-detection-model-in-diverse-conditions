# AGC_final

### inference code
1. download the pre-trained .pth weight files in the save_models. There are many pre-trained models for running the code.
2. run the prediction code, and specify your own video dataset directory with pre-defined.json. We provide the sample video.mp4 and pre-defined.json files.
'''bash
python main.py --video_root_dir sample_dataset
'''
3. You can see the predicted json files in the ./debug/answer.
4. You can also see the first ocr prediction results in the ./debug/first_ocr_results
