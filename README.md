# Coursera_DL0110EN_Honors : Fashion-MNIST Project
Coursera_DL0110EN_Honors : Fashion-MNIST Project

## Create a custom class for dataset

👧💬 🎈 We had been talking about this before but the custom dataset can contain of data frame, specific method for data loading and management, functions to generated data, API Key or communication method that allows you to access the current update online of the data or transform functions. </br>
🦭💬 In previous versions fixed number of data stored in the dataset and they generate of the data by the target number of records estimation, working on some exames you need to verify of the data and drop of null values that is intention of the instructors. Seek and scan the data before summation and verify matching data types before comparison. </br>
🦤💬 Why I have to update everytime I using in new computer even on my laptop, anyway I have an off-line versions from my student's friend. </br>
👨🏻‍🏫(1)💬 It is working but you need to active anti-virus. </br>
👨🏻‍🏫(2)💬 We are working on RMS that is because it is proven linearly. </br>
🐱💬 🎵🎶 I known of you both and not try to prove anything anyone have a paid job for me⁉️ </br> 
</br>
🐑💬 ➰ We are talking about what we can working with the custom dataset, in category dataset we can read and transform of data or create output from the dataset object at the specific index or criteria without using the database buffer if you working with a custom dataset or you can using HDFS or FS with the common dataset you do not work with a custom dataset class. </br>

```
# 🧸💬 Create data set, as Dataset and DataLoader in Pythorch
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None, ratios=0.5):
        
        # Image directory
        self.data_dir=data_dir
        
        # The transform is goint to be used on image
        self.transform = transform
        data_dircsv_file=os.path.join(self.data_dir,csv_file)
        # Load the CSV file contians image info
        self.data_name= pd.read_csv(data_dircsv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0] 
        
        # 🧸💬 to create training and validate dataset 
        self.ratios = ratios
        self.max_train_index = int( 100 * ( 1 - self.ratios ) )
        self.max_validate_index = 100 - self.max_train_index
    
    def __test__(self):
        
        print( self.data_name.iloc[0, 0] )
        return
    
    # Get the length
    def __len__(self):
#         return self.len
        # 🧸💬 incorrect length number
        return 100
        
        # Getter
    def __getitem__(self, idx):
        
        # Image file path
        img_name=os.path.join(self.data_dir,self.data_name.iloc[idx, 1])
        # Open image file
        image = Image.open(img_name)
        
        # The class label for the image
        y = self.data_name.iloc[idx, 0]
        
        # 🧸💬 Category condition or pre-defined functions.
        # set y value as int32
        if y == "Ankle boot" :
            y = 0
            
        # set y value as int32
        elif y == "T-shirt" :
            y = 1
            
        # set y value as int32
        elif y == "Coat" :
            y = 2
            
        # set y value as int32
        elif y == "Dress" :
            y = 3
            
        # set y value as int32
        elif y == "Trouser" :
            y = 4
            
        # set y value as int32
        elif y == "Pullover" :
            y = 5
            
        # set y value as int32
        elif y == "Shirt" :
            y = 6
            
        # set y value as int32
        elif y == "Sandal" :
            y = 7
            
        # set y value as int32
        elif y == "Sneaker" :
            y = 8
            
        # set y value as int32
        elif y == "Bag" :
            y = 9
            
        else:
            print( y )

        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)
            
        return image, y
```

<p align="center" width="100%">
    <img width="100%" src="https://github.com/jkaewprateep/Coursera_DL0110EN_Honors/blob/main/1.png">
</p>

## Image dataset and custom variable stored into TensorFlow logging and database

<p align="center" width="100%">
    <img width="100%" src="https://github.com/jkaewprateep/Coursera_DL0110EN_Honors/blob/main/4.png">
</p>

- - - 

## Create a custom class for layer

```
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
        
    def build(self, input_shape):
        min_size_init = tf.keras.initializers.RandomUniform(minval=10, maxval=10, seed=None)
        self.kernel = self.add_weight(shape=[int(input_shape[-1]),
                        self.num_outputs],
                        initializer = min_size_init,
                        trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
```

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class MyLSTMLayer( tf.keras.layers.LSTM ):
    def __init__(self, units, return_sequences, return_state):
        super(MyLSTMLayer, self).__init__( units, return_sequences=True, return_state=False )
        self.num_units = units

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
        shape=[int(input_shape[-1]),
        self.num_units])

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
```

<p align="center" width="100%">
    <img width="100%" src="https://github.com/jkaewprateep/Coursera_DL0110EN_Honors/blob/main/2.png">
</p>

- - -

## Create a custom class for function

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""   
def normal_sp(params):
    return tfd.Normal(loc=params,\
                      scale=1e-5 + 0.00001*tf.keras.backend.exp(params))# both parameters are learnable
```

<p align="center" width="100%">
    <img width="100%" src="https://github.com/jkaewprateep/Coursera_DL0110EN_Honors/blob/main/3.png">
</p>

- - -

## Create a learning as criterion with Gradient Tape

```
import time
start_time = time.time()

cost_list=[]
accuracy_list=[]
N_test=len(dataset_val)
# n_epochs=5
n_epochs=10
for epoch in range(n_epochs):
    cost=0
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        z = model(x)
#         loss = criterion(z, y)
        loss = criterion(z, torch.tensor(y))
        loss.backward()
        optimizer.step()
        cost+=loss.item()
    correct=0
    
    #perform a prediction on the validation  data 
    model.eval()
    for x_test, y_test in test_loader:
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
    accuracy = correct / N_test
    accuracy_list.append(accuracy)
    cost_list.append(cost)
```

<p align="center" width="100%">
    <img width="100%" src="https://github.com/jkaewprateep/Coursera_DL0110EN_Honors/blob/main/5.png">
</p>

- - -

## Matrix weights response distribution

```
# 🧸💬 Cross-entrophy loss
criterion = nn.CrossEntropyLoss()

def rms_criterion(yhat,y):
    out = -1 * torch.mean(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))
    return out
```

<p align="center" width="100%">
    <img width="100%" src="https://github.com/jkaewprateep/Coursera_DL0110EN_Honors/blob/main/6.png">
</p>

[DekDee]( https://stackoverflow.com/users/7848579/jirayu-kaewprateep )
