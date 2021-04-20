import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def compute_convolution(I, T, stride=10):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    #template = Image.open('C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw1_data/red_light_inputs/red_light_temp_crop5.jpg')
    #temp_arr = np.asarray(template)
    temp = T
    
    #heatmap = np.random.random((n_rows, n_cols))
    if (stride == None):
        stride = 10
        
    dim = np.shape(I[:,:,0])
    temp_dim = np.shape(temp[:,:,0])
    dots = np.empty((int(n_rows/stride), int(n_cols/stride)))
    dots[:] = np.nan
    #heat = [[],[],[]] #int(n_rows/(stride))
    for color in range(3):
        column = 0
        rcol = 0
        temp_vec = np.reshape(temp[:,:,color], (-1,))
        #temp_vec = temp_vec/(max(temp_vec)/2) -1
        while column < (dim[1]-temp_dim[1]):
            # scan across rows
            row = 0
            rheat = 0
            while row < (dim[0] - len(temp[:,0,0])):
                #box = [row, column, row + len(temp[:,0,0]) , column + len(temp[0,:,0])]
                sample = I[row:(row + len(temp[:,0,color])), column:(column + len(temp[0,:,color])), color]
                sample_vec = np.reshape(sample, (-1,))
                #sample_vec = sample_vec/(max(sample_vec)/2) -1
                #print(np.shape(sample_vec))
                #print(np.shape(temp_vec))
                dot = np.convolve(sample_vec, temp_vec, 'valid')
                #print(dot)

                dots[rheat, rcol] = dot
                #print(dots)
                row += stride
                rheat += 1
            column += stride
            rcol += 1
            #print(dots)
        dots_clean = dots[~np.isnan(dots).all(axis=1)]
        #print(np.shape(dots_clean))
        dots_clean = dots_clean.T[~np.isnan(dots_clean.T).all(axis = 1)].T
        #print(np.shape(dots_clean))
        if (color == 0):
            heat0 = dots_clean
        elif (color == 1):
            heat1 = dots_clean
        else:
            heat2 = dots_clean

    #print(heat0)
    #print(heat1)
    #print(heat2)
    #heat = [[heat0], [heat1], [heat2]]
    #print(np.shape(heat))
    
    
    heat = (heat0 + heat1 + heat2)/3
    
    #img = Image.fromarray(np.array(heat))
    
    #img.show()
    #print(np.max(heat))
    heatmap = heat
    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''
    stride = 10

    output = []

    '''
    BEGIN YOUR CODE
    '''
    temp_dim = [49,26]
    tempR = 188 #from convolving temp channels with itself
    tempB = 68
    tempG = 164
    
    #print(np.shape(heatmap))
    #print(np.shape(heatmap[0]))
    #print(np.max(heatmap[0]))
    # normalize respective channels using above numbers
    #heatmap[0] = heatmap[0]/np.max(heatmap[0])
    #heatmap[1] = heatmap[1]/np.max(heatmap[1])
    #heatmap[2] = heatmap[2]/np.max(heatmap[2])
    #heat = (heatmap[0] + heatmap[1] + heatmap[2])/3
    #print(np.shape(heat))
    heat = heatmap/np.max(heatmap)
    for  row in np.arange(np.shape(heat)[0]):
        for col in np.arange(np.shape(heat)[1]):
            item = heat[row, col]
            if (item > 0.7) and (item < 1):
                ind = [row, col]
                if (ind[0] == 0):
                    tl_row = 0
                else:
                    tl_row = ind[0]*stride
                if (ind[1] == 0):
                    tl_col = 0
                else:
                    tl_col = ind[1]*stride
                br_row = int(tl_row + temp_dim[0])
                br_col = int(tl_col + temp_dim[1])
                score = float(item)
                output.append([int(tl_row),int(tl_col),br_row,br_col, score])    
    
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    '''
    box_height = 8
    box_width = 6

    num_boxes = np.random.randint(1,5)

    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)

        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width

        score = np.random.random()

        output.append([tl_row,tl_col,br_row,br_col, score])
    '''
    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    #template_height = 8
    #template_width = 6

    # You may use multiple stages and combine the results
    template = Image.open('C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw1_data/red_light_inputs/red_light_temp_crop5.jpg')
    temp_arr = np.asarray(template)
    T = temp_arr #np.random.random((template_height, template_width))

    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap)
    #print(type(output[0][3]))

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw2_data/RedLights2011_Medium'

# load splits: 
split_path = 'C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw2_data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw2_data/hw02_preds' 
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)
    
    ################### visualize ############################
    #plt.imshow(I)
    #boxes = preds_train[file_names_train[i]][:4]
    #for box in boxes:
        #plt.gca().add_patch(Rectangle((box[1], box[0]), 26,49, fill = False, color="purple",
                       #linewidth=2))
    #plt.show()
    ###########################################################   

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)


    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
