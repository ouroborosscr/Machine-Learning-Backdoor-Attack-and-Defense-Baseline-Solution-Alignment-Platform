from __future__ import absolute_import, division, print_function 



from GradEst.hsja import hsja
import numpy as np
import tensorflow as tf

import os






def attack(model,basic_imgs,target_imgs,target_labels):



	vec=np.empty((basic_imgs.shape[0]))
	vec_per=np.empty_like(basic_imgs)
	model=model

	print(vec.shape)

	a=[]



	for i in range(basic_imgs.shape[0]):

		sample=basic_imgs[i]

		#
		target_image =target_imgs[i]

		target_label=target_labels[i]

		print('attacking the {}th sample...'.format(i))

		dist,per = hsja(model,
							sample, 
							clip_max = 1, 
							clip_min = 0, 
							constraint = "l2",
							num_iterations = 50,
							gamma = 1.0, 
							target_label = target_label,
							target_image = target_image,
							stepsize_search ='geometric_progression',
							max_num_evals = 3e4,
							init_num_evals = 100)


		# print(per.shape)
		vec[i]=np.max(np.abs(per - sample)) / dist
		print(vec[i])
		vec_per[i]=per-sample
		# ratio=
		# print(ratio)
		#
		# a.append(ratio)
		# print(dist)

	assert vec.all()>=0, print("GG need larger than 0")
	print(f"RATIO: {a}")
	return vec,vec_per

