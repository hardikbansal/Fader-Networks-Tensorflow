import numpy as np

def flat_batch(input_batch, batch_size, final_width, final_height):
	if(final_height*final_width != batch_size):
		print("Error flattening batch size. batch_size != final_width*final_height")
	else:
		h = 256
		w = 256
		output = np.zeros((final_width*h, final_height*w))

		for idx, image in enumerate(input_batch):
			i = idx % final_width
			j = int(idx / final_height)
			output[j*h:j*h+h, i*w:i*w+w] = image.reshape((w,h))

	return output
