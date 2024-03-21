# Coursera_DL0110EN_Honors : Fashion-MNIST Project
Coursera_DL0110EN_Honors : Fashion-MNIST Project

## Create a custom class for dataset

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
