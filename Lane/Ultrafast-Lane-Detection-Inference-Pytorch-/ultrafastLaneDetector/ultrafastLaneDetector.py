import cv2
import torch
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist
from ultrafastLaneDetector.model import parsingNet

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

class ModelConfig():
	def __init__(self, model_type):
		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18

class UltrafastLaneDetector():
	def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):
		self.use_gpu = use_gpu
		self.cfg = ModelConfig(model_type)
		self.model = self.initialize_model(model_path, self.cfg, use_gpu)
		self.img_transform = self.initialize_image_transform()

	@staticmethod
	def initialize_model(model_path, cfg, use_gpu):
		net = parsingNet(pretrained=False, backbone='18', cls_dim=(cfg.griding_num+1, cfg.cls_num_per_lane, 4), use_aux=False)
		if use_gpu:
			if torch.backends.mps.is_built():
				net = net.to("mps")
				state_dict = torch.load(model_path, map_location='mps')['model']
			else:
				net = net.cuda()
				state_dict = torch.load(model_path, map_location='cuda')['model']
		else:
			state_dict = torch.load(model_path, map_location='cpu')['model']

		compatible_state_dict = {k[7:] if 'module.' in k else k: v for k, v in state_dict.items()}
		net.load_state_dict(compatible_state_dict, strict=False)
		net.eval()
		return net

	@staticmethod
	def initialize_image_transform():
		return transforms.Compose([
			transforms.Resize((288, 800)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

	def detect_lanes(self, image, draw_points=True):
		input_tensor = self.prepare_input(image)
		output = self.inference(input_tensor)
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)
		visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)
		return visualization_img, self.lanes_points, self.lanes_detected, self.cfg, draw_points

	def prepare_input(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_pil = Image.fromarray(img)
		input_img = self.img_transform(img_pil)
		input_tensor = input_img[None, ...]
		if self.use_gpu and not torch.backends.mps.is_built():
			input_tensor = input_tensor.cuda()
		return input_tensor

	def inference(self, input_tensor):
		with torch.no_grad():
			output = self.model(input_tensor)
		return output

	@staticmethod
	def process_output(output, cfg):
		processed_output = output[0].data.cpu().numpy()
		processed_output = processed_output[:, ::-1, :]
		prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
		idx = np.arange(cfg.griding_num) + 1
		idx = idx.reshape(-1, 1, 1)
		loc = np.sum(prob * idx, axis=0)
		processed_output = np.argmax(processed_output, axis=0)
		loc[processed_output == cfg.griding_num] = 0
		processed_output = loc

		col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
		col_sample_w = col_sample[1] - col_sample[0]

		lanes_points = []
		lanes_detected = []

		max_lanes = processed_output.shape[1]
		for lane_num in range(max_lanes):
			lane_points = []
			if np.sum(processed_output[:, lane_num] != 0) > 2:
				lanes_detected.append(True)
				for point_num in range(processed_output.shape[0]):
					if processed_output[point_num, lane_num] > 0:
						lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)
			lanes_points.append(lane_points)

		# Đảm bảo các phần tử trong lanes_points có cùng hình dạng
		max_points = max(len(lane) for lane in lanes_points)
		uniform_lanes_points = [lane + [[-1, -1]] * (max_points - len(lane)) for lane in lanes_points]

		return np.array(uniform_lanes_points), np.array(lanes_detected)

	@staticmethod
	def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
		visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_AREA)
		if lanes_detected[1] and lanes_detected[2]:
			lane_segment_img = visualization_img.copy()
			cv2.fillPoly(lane_segment_img, pts=[np.vstack((lanes_points[1], np.flipud(lanes_points[2])))], color=(255, 191, 0))
			visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

		if draw_points:
			for lane_num, lane_points in enumerate(lanes_points):
				for lane_point in lane_points:
					if lane_point[0] >= 0 and lane_point[1] >= 0:
						cv2.circle(visualization_img, (lane_point[0], lane_point[1]), 3, lane_colors[lane_num], -1)
		return visualization_img
