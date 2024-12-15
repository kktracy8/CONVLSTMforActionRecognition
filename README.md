# CONVLSTMforActionRecognition

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
