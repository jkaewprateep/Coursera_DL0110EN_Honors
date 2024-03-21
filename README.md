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
🐐💬 I can add a transformation function, one-hot-vector or random generate output for custom dataset for games and simulations but custom dataset need to be defined because custom functions, custom layer are working on local machines when datasets generated of data you need to perform at training iterations. Thinking about library database, ABCD colour and alphabet matching games 🟥🟨🟩🅰️🅱️ and reserach data they are working as database class and you can validate of data or transform data into multiple formats ISBN for books or label for laboratory objects. </br>
🥺💬 I think to have kids they need to have some stable resources my dad working at the convenience store create a barcode reader and accounting program for buy me a computer too. 🪣🐡🐙🪸🐳 </br>

```
IMAGE_SIZE = 16
transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                 # 🧸💬 Transfrom function you can add augmentations for the output.

# 🧸💬 Create data set, as Dataset and DataLoader in Pythorch
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):                                                           # 🧸💬 Create a simple calss definition.

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None, ratios=0.5):           # 🧸💬 Python class initial loader __init__ .
        
        # Image directory
        self.data_dir=data_dir                                                    # 🧸💬 Initial internal class variable.
        
        # The transform is goint to be used on image
        self.transform = transform                                                # 🧸💬 Initial internal class variable.
        data_dircsv_file=os.path.join(self.data_dir,csv_file)
        # Load the CSV file contians image info
        self.data_name= pd.read_csv(data_dircsv_file)                             # 🧸💬 Download data from target .csv file.
        
        # Number of images in dataset
        self.len=self.data_name.shape[0]                                          # 🧸💬 Initial internal class variable.
        
        # 🧸💬 to create training and validate dataset 
        self.ratios = ratios                                                      # 🧸💬 Ratios determine the selection number.
        self.max_train_index = int( 100 * ( 1 - self.ratios ) )                   # 🧸💬 Conversion to int number.
        self.max_validate_index = 100 - self.max_train_index                      # 🧸💬 Remainder records for testing input.
    
    def __test__(self):
        
        print( self.data_name.iloc[0, 0] )                                        # 🧸💬 Validate result, __init__ load success.
        # 🐑💬 ➰ If __init__ load success not only return of the function but the first record or selection record have correct shape.
        # 🧸💬 This is a bug in 0 return function and some programmer used to use -1 or 0 return or return logics from APIs.
        return                                                                    
    
    # Get the length
    def __len__(self):
#         return self.len                                                         # 🧸💬 Create target numbers or ratios for the selection dataset
        # 🧸💬 incorrect length number                                           # 🧸💬 Method overriding can perform but by programming manners, you
        return 100                                                                # 🧸💬 need to specify of the selection method as input because users
                                                                                  # 🧸💬 are not reading though the custom class even readme.txt provided.
        # Getter
    def __getitem__(self, idx):                                                   # 🧸💬 Select images by the reference of the .csv file.
        
        # Image file path
        img_name=os.path.join(self.data_dir,self.data_name.iloc[idx, 1])          # 🧸💬 Reading an image from a file path, memory has only index of items.
                                                                                  # 🦤💬 Am I hearing some of these requirements anywhere ⁉️
                                                                                  # 🥺💬 They keep negative votes on my answers then I answer here.
        # Open image file
        image = Image.open(img_name)                                              # 🧸💬 image loading method from Image library.
        
        # The class label for the image
        y = self.data_name.iloc[idx, 0]                                           # 🧸💬 Initial of dataset record label assigned to a local variable.
        
        # 🧸💬 Category condition or pre-defined functions.
        # set y value as int32
        if y == "Ankle boot" :                                                    # 🧸💬 Expanding condition to present of matching or transform you can
            y = 0                                                                 # 🧸💬 work in the dataset.
            
        # set y value as int32
        elif y == "T-shirt" :                                                     # 🧸💬 Expanding condition to present of matching or transform you can
            y = 1                                                                 # 🧸💬 work in the dataset.
            
        # set y value as int32
        elif y == "Coat" :                                                        # 🧸💬 Expanding condition to present of matching or transform you can
            y = 2                                                                 # 🧸💬 work in the dataset.
            
        # set y value as int32
        elif y == "Dress" :                                                       # 🧸💬 Expanding condition to present of matching or transform you can
            y = 3                                                                 # 🧸💬 work in the dataset.
            
        # set y value as int32
        elif y == "Trouser" :                                                     # 🧸💬 Expanding condition to present of matching or transform you can
            y = 4                                                                 # 🧸💬 work in the dataset.
            
        # set y value as int32
        elif y == "Pullover" :                                                    # 🧸💬 Expanding condition to present of matching or transform you can
            y = 5                                                                 # 🧸💬 work in the dataset.
            
        # set y value as int32
        elif y == "Shirt" :                                                       # 🧸💬 Expanding condition to present of matching or transform you can
            y = 6                                                                 # 🧸💬 work in the dataset.
            
        # set y value as int32
        elif y == "Sandal" :                                                      # 🧸💬 Expanding condition to present of matching or transform you can
            y = 7                                                                 # 🧸💬 work in the dataset.
            
        # set y value as int32
        elif y == "Sneaker" :                                                     # 🧸💬 Expanding condition to present of matching or transform you can
            y = 8                                                                 # 🧸💬 work in the dataset.
            
        # set y value as int32
        elif y == "Bag" :                                                         # 🧸💬 Expanding condition to present of matching or transform you can
            y = 9                                                                 # 🧸💬 work in the dataset.
            
        else:
            print( y )                                                            # 🧸💬 Something else not matching return the same, working with future
                                                                                  # 🧸💬 category.

        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)                                         # 🧸💬 Transform an image by transform function for output.
            
        return image, y
```

<p align="center" width="100%">
    <img width="100%" src="https://github.com/jkaewprateep/Coursera_DL0110EN_Honors/blob/main/1.png">
</p>

## Image dataset and custom variable stored into TensorFlow logging and database

💃( 👩‍🏫 )💬 I have some requirements about ``` How do I create aggregated reports from labels input prediction or environments input values when there is some simulation and need to create a single report and dashboard from the TensorFlow engine with benefits of logging, memory, disk space and software versions management? ``` </br>
🦤💬 Are you talking about working records backlog tracing? On the day we are talking about this issue for explain of selection category from a list of actions and limited of actions performing to ```evaluation of the method```, compare to the experience this can be tools support of data input and data fields selection and the ability of development of the programmer, business units and computer. </br>
🦤💬 Data extraction from the source can be imported and ```working with system solution without number of merged times```, you can add of supervised label or scores or advisory scores for the supervised learning method. </br>
💃( 👩‍🏫 )💬 It is ```backward tracing compatibilities```, as in example you create password from generated method provide and you can input of parameters for target number of distributes value result from the method and you may need to have recognition number as configured parameters and validated numbers for backtracing when there is some audits seasons the working of the function and method are provide the ```acceptable range from the inputs without retest``` the whole system. </br>


<p align="center" width="100%">
    <img width="100%" src="https://github.com/jkaewprateep/Coursera_DL0110EN_Honors/blob/main/4.png">
</p>

- - - 

## Create a custom class for layer

🦤💬 I give some assignment to a student for SoftMax layer, Normalized layer ... </br>
🥺💬 I answered about 5 years agos, now see more example people asking on the StackOverflow they are more desired on implement architecture and supporting of some specific system for OCR and documentation modules ... </br>
🐑💬 ➰ There is no payback money or coins but that is time spending and working experience shared from study and experiments when a new comer can search and implement of the guideline method with rocket 🚀 </br>

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class MyDenseLayer(tf.keras.layers.Layer):                                          # 🧸💬 Create a custom class as Layer type.
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()                                        # 🧸💬 Create a new class object with the same type.
                                                                                    # 🦭💬 Prevent shared memory problem 👨🏻‍🏫(2)💬 .
        self.num_outputs = num_outputs                                              # 🧸💬 Initial number of output assign a local variable.
        
    def build(self, input_shape):                                                   # 🧸💬 Initail weights and shape of networks kernel.
        min_size_init = tf.keras.initializers.RandomUniform(minval=10, maxval=10, seed=None)
        self.kernel = self.add_weight(shape=[int(input_shape[-1]),
                        self.num_outputs],
                        initializer = min_size_init,
                        trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)                                      # 🧸💬 Calculation outputs.
```

### 🧸💬 Create a custom LSTM layer class.

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class MyLSTMLayer( tf.keras.layers.LSTM ):                                         # 🧸💬 Create a custom class as Layer type.
    def __init__(self, units, return_sequences, return_state):
        super(MyLSTMLayer, self).__init__( units, return_sequences=True, return_state=False )
        self.num_units = units

    def build(self, input_shape):                                                  # 🧸💬 Initial number of output assign a local variable.
        self.kernel = self.add_weight("kernel",
        shape=[int(input_shape[-1]),
        self.num_units])

    def call(self, inputs):
        # 🧸💬 Replace with LSTM layer from TensorFlow layer.                     # 🧸💬 Calculation outputs.
        return tf.matmul(inputs, self.kernel)
```

<p align="center" width="100%">
    <img width="100%" src="https://github.com/jkaewprateep/Coursera_DL0110EN_Honors/blob/main/2.png">
</p>

- - -

## Create a custom class for function

🐨🎁🎵🎶 A custom function, ``` In technology fields, how do our student retain ability and potential when settings and use cases of the equipment are the same⁉️ ``` </br>
👨🏻‍🏫💬 We can teach them to create custom functions, and semi-functions that help in multiple steps of learning and practice to use in the actual cases. </br>
👧💬 🎈 I go to the exhibitions and watch the equipment use cases when my friends do not understand of the ```import/export``` modes. </br>
👨🏻‍🏫💬 Trust me we are working on fields and research it is for the situation and Thai students. </br>
🦭💬 One of them is the designers they speak on television, have anyone try visit my exhibitions⁉️ </br>

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

💃( 👩‍🏫 )💬 This is working as the TensorFlow version 1, and in Pythorch we can do the same they combined of the loss estimation function with historical logging and sometimes matrixes estimators as criterions. </br>
🦤💬 In a large scale computation we still use of this pattern you can see from the Gradient method, they are working on communication part more than calculation part as they are specifically use on the distributed system and when the question is do we need to use a server for none distribution system or specific calculation units this is the answer. </br> 

```
import time
start_time = time.time()                                                            # 🧸💬 Initial start time variable.                                                      

cost_list=[]                                                                        # 🧸💬 Initial cost array variable.
accuracy_list=[]                                                                    # 🧸💬 Initial accuracy array variable.
N_test=len(dataset_val)                                                             # 🧸💬 Initial length of validation dataset records.
n_epochs=10                                                                         # 🧸💬 Initial number of epoaches setting.
for epoch in range(n_epochs):                                                       # 🧸💬 Define iterations learning.
    cost=0                                                                          # 🧸💬 Initial cost variable.
    model.train()                                                                   # 🧸💬 Flagged training state to the model.
    for x, y in train_loader:                                                       # 🧸💬 Iterates though data loader records.
        optimizer.zero_grad()                                                       # 🧸💬 Initial optimizer with initial value 0.
        z = model(x)                                                                # 🧸💬 Define a model.                                           
        loss = criterion(z, torch.tensor(y))                                        # 🧸💬 Find loss value estimates for training decission.
        loss.backward()                                                             # 🧸💬 Loss estimation value output.
        optimizer.step()                                                            # 🧸💬 Continue training by step function.
        cost+=loss.item()                                                           # 🧸💬 Appending cost array with loss items output.
    correct=0                                                                       # 🧸💬 Initial variable by value 0.
    
    #perform a prediction on the validation  data 
    model.eval()                                                                    # 🧸💬 Evaluation method 👨🏻‍🏫💬 I am not puty I play PyTorch 🥲 🦭💬 ... 
    for x_test, y_test in test_loader:                                              # 🧸💬 Define iterations for test_data from data loader.
        z = model(x_test)                                                           # 🧸💬 Prediction function without test is multiplication.
        _, yhat = torch.max(z.data, 1)                                              # 🧸💬 Select prediction result from max possibility values.
        correct += (yhat == y_test).sum().item()                                    # 🧸💬 Couting number of correct values.
    accuracy = correct / N_test                                                     # 🧸💬 Define a calculation accuracy values.
    accuracy_list.append(accuracy)                                                  # 🧸💬 Appending accuracy value to accuracy list.
    cost_list.append(cost)                                                          # 🧸💬 Appending cost value to cost list.
```

<p align="center" width="100%">
    <img width="100%" src="https://github.com/jkaewprateep/Coursera_DL0110EN_Honors/blob/main/5.png">
</p>

- - -

## Matrix weights response distribution

🐐💬 Loss estimation functions or estimating value function is simply and you can create by using simple of calculation method with logarithms scales or time scales but background they had mathametics you may consider since number of dimension or observations increase the number of inputs is more than two or three inputs possibility more. Remain the matrixes sizes dimension you need to put some intelligence sometimes is expreinces and mathematical for the problem sovlers </br> 

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
