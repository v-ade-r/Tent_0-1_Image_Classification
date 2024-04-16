# Tent_0-1_Image_Classification
Fine-tuned EfficientNetV2B0, for binary Image_Classification, with Gradio.

I only had 48 images of my tent, and I was wondering if that would be enough to fine-tune EfficientNetV2B0, after augmenting the data slightly. I decided to do it using cv2 for fun. 

There are 2 labels: **Tent, NonTent**. Their ratio is almost 50/50. Number of images after data augmentation:
1. **Train** data size: **504 files**
2. **Valid** data size: **112 files**
3. **Test** data size: **12 files**

You can learn from this code how to:

1. **Augment the data** with cv2
2. **Prepare the data** for the model
3. Quickly **fine-tune the mode**
4. **Use Gradio**

**Conclusions:**
This fine-tuned model was trained solely on images of my white MSR Hubba Hubba NX tent, but it still managed to learn the essence of ‘tentness’ and learn how to recognize a tent of any color and on any type of ground, not just grass. The percentages are usually lower for very difficult setups, but the predictions are still correct most of the time. Moreover, it can differentiate between tents and similar objects like houses, barns, and igloos.

**Final utility**
This code enables any user visiting the site (new address is created by Gradio during each compilation.) to drag or upload any image and get an evaluation of whether there is a tent somewhere in the image or not. The percentages given next to the label explain how confident the model is in its prediction.
