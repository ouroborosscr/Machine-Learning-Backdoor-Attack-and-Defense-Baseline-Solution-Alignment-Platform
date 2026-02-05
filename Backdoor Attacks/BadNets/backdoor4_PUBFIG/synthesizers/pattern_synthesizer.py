#验证修后ASR效果，不选
import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class PatternSynthesizer(Synthesizer):
    pattern_tensor: torch.Tensor = torch.load("synthesizers/Trojan_Square_10hid")
    # pattern_tensor: torch.Tensor = torch.tensor([
    #     [1., 0., 1., 0., 1.],
    #     [0., 1., 0., 1., 0.],
    #     [1., 0., 1., 0., 1.],
    #     [0., 1., 0., 1., 0.],
    #     [1., 0., 1., 0., 1.]
    # ])
    "Just some random 2D pattern."

    x_top = 0
    "X coordinate to put the backdoor into."
    y_top = 0
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    resize_scale = (5, 10)
    "If the pattern is dynamically placed, resize the pattern."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, task: Task):
        super().__init__(task)
        self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)

    def make_pattern(self, pattern_tensor, x_top, y_top):
        # print(len(list(pattern_tensor.shape)))
        # die()
        if(len(list(pattern_tensor.shape))==2):
            full_image = torch.zeros(self.params.input_shape)
            full_image.fill_(self.mask_value)

            x_bot = x_top + pattern_tensor.shape[0]
            y_bot = y_top + pattern_tensor.shape[1]
            
            if x_bot > self.params.input_shape[1] or \
                    y_bot > self.params.input_shape[2]:
                raise ValueError(f'Position of backdoor outside image limits:'
                                 f'image: {self.params.input_shape}, but backdoor'
                                 f'ends at ({x_bot}, {y_bot})')

            full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor
            

            self.mask = 1 * (full_image != self.mask_value).to(self.params.device)

            self.pattern = self.task.normalize(full_image).to(self.params.device)
        elif(len(list(pattern_tensor.shape))==3):
            full_image = torch.zeros(self.params.input_shape)
            full_image.fill_(self.mask_value)

            x_bot = x_top + pattern_tensor.shape[1]
            y_bot = y_top + pattern_tensor.shape[2]



            if x_bot > self.params.input_shape[1] or \
                    y_bot > self.params.input_shape[2]:
                raise ValueError(f'Position of backdoor outside image limits:'
                                 f'image: {self.params.input_shape}, but backdoor'
                                 f'ends at ({x_bot}, {y_bot})')
            #print(pattern_tensor.shape)

            #full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor
            full_image[0, x_top:x_bot, y_top:y_bot] = pattern_tensor[0]
            full_image[1, x_top:x_bot, y_top:y_bot] = pattern_tensor[1]
            full_image[2, x_top:x_bot, y_top:y_bot] = pattern_tensor[2]

            self.mask = 1 * (full_image != self.mask_value).to(self.params.device)

            self.pattern = self.task.normalize(full_image).to(self.params.device)

        else:
            print("'pattern_tensor' only need 1 or 3 batch")
            die()

    def synthesize_inputs(self, batch, attack_portion=None):
        pattern, mask = self.get_pattern()
        batch.inputs[:attack_portion] = (1 - mask) * \
                                        batch.inputs[:attack_portion] + \
                                        mask * pattern

        return

    def synthesize_labels(self, batch, attack_portion=None):
        
        # batch.labels[:attack_portion].fill_(self.params.backdoor_label)
        # print(batch.labels)
        # print(batch.labels.tolist())
        for i in range(len(batch.labels.tolist())):
            if(batch.labels[i]==6):
                batch.labels[i]=0
            else:
                batch.labels[i]=6
        # print(batch.labels)
        # die()



        return

    def get_pattern(self):
        if self.params.backdoor_dynamic_position:
            resize = random.randint(self.resize_scale[0], self.resize_scale[1])
            pattern = self.pattern_tensor
            if random.random() > 0.5:
                pattern = functional.hflip(pattern)
            image = transform_to_image(pattern)
            pattern = transform_to_tensor(
                functional.resize(image,
                    resize, interpolation=0)).squeeze()

            x = random.randint(0, self.params.input_shape[1] \
                               - pattern.shape[0] - 1)
            y = random.randint(0, self.params.input_shape[2] \
                               - pattern.shape[1] - 1)
            self.make_pattern(pattern, x, y)

        return self.pattern, self.mask
