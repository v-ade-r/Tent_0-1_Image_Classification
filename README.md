# Tent_0-1_Image_Classification
Fine-tuned EfficientNetV2B0, for binary Image_Classification, with Gradio.

I only had 48 images of my tent, and I was wondering if that would be enough to fine-tune EfficientNetV2B0, after augmenting the data slightly. I decided to do it using cv2 for fun. This function effectively gives me 7 images (6 new)
from each single image I had before. I manually inspected it, and divided into appropriate sets, trying to maintain as much image variation as possible in each set. Nontent images have been augmented, in order to prevent the model from learning that, for example, rotated images are always a Tent.

There are 2 labels: **Tent, NonTent**. Their ratio is almost 50/50.
1. **Train** data size: **504 files**
2. **Valid** data size: **112 files**
3. **Test** data size: **12 files**

You can learn from this code how to:

1. **Augment the data** with cv2
2. **Prepare the data** for the model
3. Quickly **fine-tune the mode**
4. **Use Gradio**

**Conclusions:**
This fine-tuned model was trained solely on images of my white MSR Hubba Hubba NX tent, but it still managed to learn the essence of ‘tentness’ and can recognize a tent of any color and on any type of ground, not just grass. The percentages are usually lower, but the predictions are still correct most of the time. Moreover, it can differentiate between tents and similar objects like houses, barns, and igloos.
