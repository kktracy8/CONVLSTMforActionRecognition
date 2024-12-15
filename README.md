# CS7643

* Dimensions:
	* ConvLSTM: Loop ConvLSTMcell across each seq len (frames)
 		* Input: Batch Size, Seq Len, Input Channel (hard code = 1 for gray), H, W
   		* Hidden: Batch Size, Output Channel (desired depth of feature map), H, W
     		* Cell: Same as Hidden
       		* Output: Batch Size, Seq Len, Output Channel, H, W
         * ConvLSTMcell:
         	* Dimension details stated in forward function.
          
* The spatial dimension (height and width) after convolution operation needs to be maintained, as the gate data needs to be fixed in shape. Make sure to adjust padding accordingly for to maintain H,W after convolution with desired kernel size and stride.

* Loss: CrossEntropy in default

* GrandCam: Error for hook part, kindly please help. And also I need to use after-training parameters, which means it should be after model.eval() of the main.py, please help for how to process.

* GrandCam-Colab Version: When processing sample data, error will be generated if setting range at 9000, so I changed to 887, the largest number allowed. Output label is not the name of behavior anymore but behaivor's order, so the title of each output raw sample image is in number. Model can't work out either, and the detail error is in the juptyer notebook.
