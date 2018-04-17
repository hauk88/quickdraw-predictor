import numpy as np;

class QdDataset:

	def __init__(self, images = 100, split = 0.8, seed = 1, path="..\\quickdraw dataset\\"):
		self.path = path;
		self.lables = ["cow", "fork", "lightning"];
		self.nrOfImages = images;
		self.trainSplit = split;
		self.trainAmount = int(self.nrOfImages*self.trainSplit);
		self.testAmount = int(self.nrOfImages-self.trainAmount);
		self.train = ImgDataset();
		self.test = ImgDataset();
		self.randomSeed = seed;
		
	def read_data(self):
		np.random.seed(self.randomSeed);
		for i in range(len(self.lables)):
			oneHotLable = np.zeros(3);
			oneHotLable[i] = 1;
			
			s = self.path + "full_numpy_bitmap_{}.npy".format(self.lables[i]);
			print(s);
			img = np.load(self.path + "full_numpy_bitmap_{}.npy".format(self.lables[i]));
			totalNrOfImages,nrOfPixels = img.shape;

			if(totalNrOfImages < self.nrOfImages):
				raise ValueError("File provided did not have enough images");

			testImages = np.zeros((self.testAmount,nrOfPixels));
			choises = [k for k in range(totalNrOfImages)];
			for j in range(self.testAmount):
				index = choises.pop(np.random.randint(0,len(choises)));
				testImages[j,] = img[index,];
			
			self.test.addData(testImages,oneHotLable);

			trainImages = np.zeros((self.trainAmount,nrOfPixels));
			for j in range(self.trainAmount):
				index = choises.pop(np.random.randint(0,len(choises)));
				trainImages[j,] = img[index,];
				
			self.train.addData(trainImages, oneHotLable);
			
	def getLable(self, lableArray):
		for i in range(len(lableArray)):
			v = lableArray[i];
			if(abs(v-1) < 10E-10):
				return self.lables[i];
				

class ImgDataset:
	def __init__(self):
		self.images = np.zeros((1,1));
		self.lables = np.zeros((1,1));
		
	def addData(self, innData, innLable):
		n = np.size(innData,0);
		duplicateLables = np.zeros((n,np.size(innLable,0)));
		i = np.nonzero(innLable)[0][0];
		duplicateLables[:,i] = np.ones((1,n));
			
		if np.size(self.images) == 1:
			self.images = innData;
			self.lables = duplicateLables;
		else:
			self.images = np.concatenate((self.images,innData),axis=0);
			self.lables = np.concatenate((self.lables,duplicateLables), axis=0);
		
	def __str__(self):
		return "Nr of images: {}, Nr of lables: {}".format(np.size(self.images,0),len(self.lables));


		
if __name__ == "__main__":
	qd = QdDataset();
	qd.read_data();
	
	print(qd.train);
	print(qd.test);
