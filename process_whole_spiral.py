import numpy as np
from PIL import Image
from scipy.ndimage import generic_filter
from skimage.transform import resize
from skimage.morphology import medial_axis, binary_closing, skeletonize 
import cv2
from skimage import img_as_bool, io, color, morphology
import matplotlib.pyplot as plt
import sys
from skimage.feature import hog
from scipy.ndimage.morphology import binary_fill_holes
from fil_finder import FilFinder2D
import astropy.units as u
import math
import optparse
from scipy.signal import butter, filtfilt
from scipy import signal
from scipy.integrate import simps
import warnings
import copy
from scipy.stats import linregress
import matplotlib.patches as patches
import os
import shelve
from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks




parser = optparse.OptionParser()



parser.add_option( '--thresh',
    action="store", dest="thresh",
    help="'Threshold value applied before binarisation", default=None)


parser.add_option( '--de',
    action="store", dest="de",
    help="Dilate and erode image before binarisation", default=None)


parser.add_option( '--ed',
    action="store", dest="ed",
    help="Erode image before binarisation", default=None)


parser.add_option( '--sc',
    action="store", dest="sc",
    help="Centre Coordinates", default=None)

parser.add_option( '--verbose',
    action="store", dest="verbose",
    help="Verbose", default=None)


parser.add_option( '--cutoff',
    action="store", dest="cutoff",
    help="Verbose", default=None)


options, args = parser.parse_args()


input_image = sys.argv[1]


# Command history to get for previously used parameters

input_base = os.path.basename(input_image)
input_base = input_base.split('.')[0]

working_dir = os.path.dirname(input_image)


command_dict  = os.path.join(working_dir,'_'.join([input_base,'command_log']))

command_dict_path  = os.path.join(working_dir,'_'.join([input_base,'command_log.db']))


pixel_mm_conversion_factor = 0.58


# Set highpass filter parameters

cutoff = (getattr(options,'cutoff'))


if not cutoff == None:
    cutoff_factor = float(cutoff)
else:
    cutoff_factor = (35/0.3)



# Get starting coordinates (if manually provided)


sc = (getattr(options,'sc'))



# Get skeleton threshold

thresh = (getattr(options,'thresh'))




if not thresh == None:
    binarize_threshold = float(thresh)
elif os.path.isfile(command_dict_path):
    command_history = shelve.open(command_dict)
    if not command_history['thresh'] == None:
        binarize_threshold = float(command_history['thresh'])
    else:
        binarize_threshold = 0.5
    command_history.close()
else:
    binarize_threshold = 0.5


verbose = (getattr(options,'verbose'))


def calc_end_points(array):
    skel_non_zero_x = np.where(array == True)[0]
    skel_non_zero_y = np.where(array == True)[1]
    end_points = []
    for i in range(len(skel_non_zero_x)):
        x = skel_non_zero_x[i]
        y = skel_non_zero_y[i]
        array_val = calc_con(array,x,y)
        if array_val < 2:
            end_points.append([x,y])
    # if len(end_points) > 2:
    #     print('Warning: ' + str(len(end_points)) + ' endpoints found')
    return end_points




def calc_forks(array):
    skel_non_zero_x = np.where(array == True)[0]
    skel_non_zero_y = np.where(array == True)[1]
    end_points = []
    for i in range(len(skel_non_zero_x)):
        x = skel_non_zero_x[i]
        y = skel_non_zero_y[i]
        array_val = calc_con(array,x,y)
        if array_val > 2:
            end_points.append([x,y])
    # if len(end_points) > 2:
    #     print('Warning: ' + str(len(end_points)) + ' endpoints found')
    return end_points



def find_nearest_endpoint(p1,endpoints,added_coords):
    distance_dict = dict()
    a = np.array(p1)
    for e in range(len(endpoints)):
        b = np.array(endpoints[e])
        if not p1 == endpoints[e] and not endpoints[e] in added_coords:
            dist = distance(a,b)
            # dist = np.linalg.norm(a-b)
            distance_dict[e] = dist
    min_index = min(distance_dict, key=distance_dict.get)
    return endpoints[min_index]




def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def intermediates(p1, p2):
    nb_points = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) - 1
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)
    return [[np.round(p1[0] + i * x_spacing), np.round(p1[1] +  i * y_spacing)] 
            for i in range(1, nb_points+1)]

def join_segments(array,starting_x,starting_y,end_points):
    added_coords = []
    added_coords.append([starting_x,starting_y])
    segment_test = 0
    processed_end_points = []
    while segment_test == 0:
        kernel_total = []
        for ix in range(starting_x-1,starting_x+2):
            for iy in range(starting_y-1,starting_y+2):
                if ix != starting_x or iy != starting_y:
                    if not [ix,iy] in added_coords:
                        array_val = array[ix][iy]
                        if array_val != 0:
                            kernel_total += (ix,iy)
        if len(kernel_total) == 2:
            starting_x = kernel_total[0]
            starting_y = kernel_total[1]
            added_coords.append([starting_x,starting_y])
        elif [starting_x,starting_y] in end_points:
            if len(end_points) - len(processed_end_points) > 2:
                closest_end_point = find_nearest_endpoint([starting_x,starting_y],end_points,added_coords)
                processed_end_points.append(closest_end_point)
                processed_end_points.append([starting_x,starting_y])
                interp_coords = intermediates([starting_x,starting_y], closest_end_point)
                for i in interp_coords:
                    if not i in added_coords:
                        interp_x = int(i[0])
                        interp_y = int(i[1])
                        array[interp_x][interp_y] = 1.0
                        added_coords.append(i)
                starting_x = int(interp_coords[-1][0])
                starting_y = int(interp_coords[-1][1])
            else:
                added_coords.append([starting_x,starting_y])
                segment_test = 1
        else:
            if len(kernel_total) > 0:
                number_in_kernel = int(len(kernel_total)/2)
                e = []
                for i in range(number_in_kernel):
                    e.append([kernel_total[2*i],kernel_total[2*i+1]])
                next_move = find_nearest_endpoint([starting_x,starting_y],e,added_coords)
                if len(next_move) == 2:
                    starting_x = next_move[0]
                    starting_y = next_move[1]
                    added_coords.append([starting_x,starting_y])
                else:
                    segment_test = 1
            else:
                segment_test = 1
    return array



def distance(P1, P2):
    return ((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2) ** 0.5

def get_centre_endpoint(array,end_points,**kwargs):
    if 'starting_coords' in kwargs:
        size_x = kwargs['starting_coords'][1]
        size_y = kwargs['starting_coords'][0]
    else:
        size_x = np.round(array.shape[0]/2)
        size_y = np.round(array.shape[1]/2)
    distance_dict = dict()
    a = np.array([size_x,size_y])
    for e in range(len(end_points)):
        b = np.array(end_points[e])
        dist = np.linalg.norm(a-b)
        distance_dict[e] = dist
    min_index = min(distance_dict, key=distance_dict.get)
    return end_points[min_index]




def return_image_kernal(array,starting_x,starting_y,added_coords,radius):
    array_shape_x = np.shape(array)[0]
    array_shape_y = np.shape(array)[1]
    kernel_total = []
    for ix in range(starting_x-radius,starting_x+(radius * 2)):
        for iy in range(starting_y-radius,starting_y+(radius * 2)):
            if ix != starting_x or iy != starting_y:
                if not [ix,iy] in added_coords:
                    if not ix >= array_shape_x and not iy >= array_shape_y:
                        array_val = array[ix][iy]
                        if array_val != 0:
                            kernel_total += (ix,iy)
    return kernel_total



def split_x_y(array,starting_x,starting_y):
    added_coords = []
    added_coords.append([starting_x,starting_y])
    segment_test = 0
    while segment_test == 0:
        radius = 1
        while radius < 5:
            kernel_total = return_image_kernal(array,starting_x,starting_y,added_coords,radius)
            if len(kernel_total) != 0:
                break
            else:
                radius = radius + 1
        if len(kernel_total) == 2:
            starting_x = kernel_total[0]
            starting_y = kernel_total[1]
            added_coords.append([starting_x,starting_y])
        elif len(kernel_total) >2:
            number_in_kernel = int(len(kernel_total)/2)
            e = []
            for i in range(number_in_kernel):
                e.append([kernel_total[2*i],kernel_total[2*i+1]])
            next_move = find_nearest_endpoint([starting_x,starting_y],e,added_coords)
            if len(next_move) == 2:
                starting_x = next_move[0]
                starting_y = next_move[1]
                added_coords.append([starting_x,starting_y])
            else:
                segment_test = 1
        else:
            segment_test = 1
    x_coords = []
    y_coords = []
    x_coords_raw = []
    y_coords_raw = []
    centre_x = added_coords[0][0]
    centre_y = added_coords[0][1]
    for a in added_coords:
        x_coords.append(a[0]-centre_x)
        y_coords.append(a[1]-centre_y)
        x_coords_raw.append(a[0])
        y_coords_raw.append(a[1])
    return x_coords,y_coords,x_coords_raw,y_coords_raw,added_coords



def calc_con(array,x,y):
    kernel_total = []
    for ix in range(x-1,x+2):
        for iy in range(y-1,y+2):
            if ix != x or iy != y:
                array_val = array[ix][iy] 
                kernel_total.append(array_val)
    kernel_val = sum(kernel_total)
    return kernel_val





def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y



def calc_radius(input_array,center_x,center_y,px,py):
    array = copy.deepcopy(input_array)
    a = np.array([px,py])
    non_zeros = []
    interp_coords = intermediates([starting_x,starting_y], [px,py])
    if len(interp_coords) > 3:
        del interp_coords[-1]
        del interp_coords[-1]
        for i in range(len(interp_coords)):
            x_pos = int(np.round(interp_coords[i][0]))
            y_pos = int(np.round(interp_coords[i][1]))
            array[x_pos][y_pos] = 1
        
        for i in interp_coords:
            array_val = calc_con(array,int(i[0]),int(i[1]))
            if array_val > 2:
                non_zeros.append(i)
        if len(non_zeros) > 0:
            closest_point = find_nearest_endpoint([px,py],non_zeros,[])
            b = np.array(closest_point)
            dist = np.linalg.norm(a-b)
        else:
             dist = None
    else:
        dist = None
    return dist



def convert_to_polar(coords,**kwargs):
    coords_radius = []
    coords_angle = []
    starting_angle = 0
    accum_angle = 0
    reductions = []
    temp_coords = copy.deepcopy(coords)
    if 'starting_coords' in kwargs:
        starting_point_x = kwargs['starting_coords'][0]
        starting_point_y = kwargs['starting_coords'][1]
    else:
        starting_point_x = temp_coords[0][0]
        starting_point_y = temp_coords[0][1]
    for i in range(len(temp_coords)):
        temp_coords[i][0] = temp_coords[i][0] - starting_point_x
        temp_coords[i][1] = temp_coords[i][1] - starting_point_y
    if 'high_range' in kwargs:
        high_range = kwargs['high_range']
    else:
        high_range = len(temp_coords)
    prev_theta = 0
    accum_angle = 0
    for i in range(high_range):
        dx = temp_coords[i][0]
        dy = temp_coords[i][1]
        radius,theta = polar(dx,dy)
        radius = radius * pixel_mm_conversion_factor
        inc_angle = theta - prev_theta
        if abs(inc_angle) > 300:
                inc_angle = theta + prev_theta
        accum_angle = accum_angle + inc_angle
        if 'verbose' in kwargs:
            # print(dx,dy)
            if not verbose == None:
                print(dx,dy,theta,inc_angle,accum_angle)
        coords_angle.append(accum_angle)
        coords_radius.append(radius)
        prev_theta = theta

    return coords_angle,coords_radius









def polar(x,y):
  return math.hypot(x,y),math.degrees(math.atan2(y,x))



def determine_spiral_direction(coords,**kwargs):
    coords_radius = []
    coords_angle = []
    starting_angle = 0
    accum_angle = 0
    starting_point_x = coords[0][0]
    starting_point_y = coords[0][1]
    theta = 0
    i = 10
    while i < len(coords):
        x = coords[i][0]
        y = coords[i][1]
        dx = int(x - starting_point_x)
        dy = int(y - starting_point_y)
        px = coords[i-1][0]
        py = coords[i-1][1]
        pdx = int(px - starting_point_x)
        pdy = int(py - starting_point_y)
        radius,theta = polar(dx,dy)
        prev_radius,prev_theta = polar(pdx,pdy)
        delta_theta = theta - prev_radius
        coords_angle.append(delta_theta)
        if 'verbose' in kwargs:
            print(i,delta_theta)
        i = i +1
    average_angle = np.mean(coords_angle)
    slope = linregress(coords_angle, range(10,i)).slope
    if average_angle < 0:
        direction = 1
    else:
        direction = 0
    return direction,slope,i



def get_sample_rate(angles,radii):
    angle_window = 20
    print(len(angles))
    point_list = []
    for i in range(len(angles)):
        points = []
        angle = angles[i]
        l = i + 1
        if l < (len(angles) - 1):
            upper_angle_diff = abs(angles[i] - angles[i])
            while upper_angle_diff < angle_window:
                upper_angle_diff = abs(angles[i] - angles[l])
                # print(i,l)
                points.append(angles[l])
                l = l + 1
                if l == (len(angles) - 1):
                    break
        sample_rate = len(points)
        point_list.append(sample_rate)
    return point_list





#########################################
#########################################



# Open image and convert to Numpy array
im = Image.open(input_image).convert('L')
im = np.array(im)


kernel = np.ones((3,3), np.uint8)


# Resize image


x_size = np.shape(im)[0]
y_size = np.shape(im)[1]


dim_ratio = x_size/y_size

new_x_dim = int(np.round(512*dim_ratio))
new_y_dim = 2048







im = resize(im, (x_size, y_size))

if not verbose == None:
    plt.imshow(im)
    plt.show()


# Threshold and binarise image

im_th = im < binarize_threshold




# Dilate and Erode image

de = (getattr(options,'de'))


if not de == None:
    print('Dilating image')
    im_th = cv2.dilate(im_th, kernel, iterations=1)
    im_th = cv2.erode(im_th, kernel, iterations=1)


ed = (getattr(options,'ed'))


if not ed == None:
    print('Eroding image')
    im_th = cv2.erode(im_th, kernel, iterations=1)
    # im_th = cv2.dilate(im_th, kernel, iterations=1)





if not verbose == None:
    plt.imshow(im_th)
    plt.show()


# Skeletonize

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    skel = skeletonize(im_th,method='lee')
    skel = np.multiply(skel, 1)

if not verbose == None:
    plt.imshow(skel)
    plt.show()


# Save result
# Image.fromarray(skel).save('result.png')


# Zero image edge

x_size = np.shape(skel)[0]
y_size = np.shape(skel)[1]

for i in range(x_size):
    skel[i][y_size-1] = 0
    skel[i][y_size-2] = 0
    skel[i][0] = 0
    skel[i][1] = 0

for i in range(y_size):
    skel[x_size-1][i] = 0
    skel[x_size-2][i] = 0
    skel[0][i] = 0
    skel[1][i] = 0

# Remove short branches

processed = morphology.remove_small_objects(skel.astype(bool), min_size=6, connectivity=2).astype(int)



with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fil = FilFinder2D(processed, distance=250 * u.pc, mask=processed)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,
    use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=2 * u.pix, prune_criteria='length')


final = fil.skeleton

if not verbose == None:
    plt.imshow(final)
    plt.show()




# Calculate number of line segments

end_points = calc_end_points(final)


# Take spiral starting point from input, or estimate from centre of all data points

if not sc == None:
    starting_coords_man = sc.split(',')
    starting_coords_man[0] = int(starting_coords_man[0])
    starting_coords_man[1] = int(starting_coords_man[1])
    starting_coords = get_centre_endpoint(final,end_points,starting_coords=starting_coords_man)
elif os.path.isfile(command_dict_path):
    command_history = shelve.open(command_dict)
    starting_coords = command_history['sc']
    command_history.close()
else:
    starting_coords = get_centre_endpoint(final,end_points)


starting_x = int(starting_coords[0])
starting_y = int(starting_coords[1])



if not verbose == None:
    print(starting_coords,binarize_threshold)


# Join Disconnected Segments

if len(end_points) > 2:
    final = join_segments(final,starting_x,starting_y,end_points)



if not verbose == None:
    plt.imshow(final)
    plt.show()


# Split curve into X and Y components



x,y,x_raw,y_raw,total = split_x_y(final,starting_x,starting_y)

middle_x = starting_x
middle_y = starting_y

# Show starting point




rect = patches.Arrow(middle_y, middle_x, 40, 30, linewidth=1, edgecolor='r', facecolor='none')

if not verbose == None:
    plt.imshow(final)
    ax = plt.gca()
    ax.add_patch(rect)
    plt.show()



# Estimate the true centre of the spiral, i.e. the point around which the spirs of each rotation are drawn around - often different from the actual start of the spiral

centre_window = 25

error_dict = dict()
coords_dict = dict()
combo=0
for tx in range(int(np.round(middle_x-centre_window)),int(np.round(middle_x+centre_window))):
    for ty in range(int(np.round(middle_y-centre_window)),int(np.round(middle_y+centre_window))):
        dummy=2
        temp_angle,temp_radius = convert_to_polar(total,starting_coords=[tx,ty])
        temp_slope = linregress(temp_angle,temp_radius).slope
        if temp_slope < 0:
            for i in range(len(temp_angle)):
                temp_angle[i] = temp_angle[i] * -1
        for i in range(len(temp_angle)):
            if temp_angle[i] >= 180:
                index_180 = i
                break
        try:
            del temp_angle[:index_180]
            del temp_radius[:index_180]
        except:
            pass

        fit_error = linregress(temp_angle,temp_radius).stderr
        error_dict[combo] = fit_error
        coords_dict[combo] = dict()
        coords_dict[combo]['x'] = tx
        coords_dict[combo]['y'] = ty
        combo = combo + 1


min_error_key = min(error_dict, key=lambda k: error_dict[k])

middle_x = coords_dict[min_error_key]['x']
middle_y = coords_dict[min_error_key]['y']


# Show centre of spiral


rect = patches.Arrow(middle_y, middle_x, 40, 30, linewidth=1, edgecolor='r', facecolor='none')

if not verbose == None:
    plt.imshow(final)
    ax = plt.gca()
    ax.add_patch(rect)
    plt.show()


# Convert from cartesian to polar coordinates


angle,radius = convert_to_polar(total,starting_coords=[middle_x,middle_y])





starting_angle = min(angle)


# Remove start of spiral - often messy


slope = linregress(angle,radius).slope

cutoff_angle = 360

if slope < 0:
    for i in range(len(angle)):
        angle[i] = angle[i] * -1


for i in range(len(angle)):
    if angle[i] >= cutoff_angle:
        index_180 = i
        break
try:
    del angle[:index_180]
    del radius[:index_180]
except:
    pass



inc_angel_difference = []
for a in range(1,len(angle)):
    diff = angle[a] - angle[a-1]
    inc_angel_difference.append(diff)



sampling_frequency = 1/np.mean(inc_angel_difference)

# Remove low frequency component





cutoff=sampling_frequency/cutoff_factor


smoothed = butter_highpass_filter(radius,cutoff,sampling_frequency)




slope = linregress(angle,radius).slope
intercept = linregress(angle,radius).intercept

fit_error_values = []

fit_error = linregress(angle,radius).stderr


lobf = []

for i in range(len(angle)):
    x = angle[i]
    y = intercept + slope*x
    true_value = radius[i]
    radius_error = abs(true_value - y)/y
    fit_error_values.append(radius_error)
    lobf.append(y)


# Calculate error


smoothed_abs = []

for i in range(len(smoothed)):
    smoothed_abs.append(abs(smoothed)[i])

mean_error = np.mean(smoothed_abs)
std_error = np.std(smoothed)
max_error = max(smoothed)



if not verbose == None:
    plt.figure(figsize=(20,10))
    plt.plot(angle,radius,color='C0',label="Raw")
    plt.plot(angle,smoothed,color='C3',label="Filtered")
    plt.plot(angle,lobf,color='C4',label="Fit")
    plt.xlabel("Incremental Angle")
    plt.ylabel("Radius")
    plt.legend(loc="upper left")
    plt.show()



if not verbose == None:
    plt.figure(figsize=(20,10))
    plt.plot(angle,smoothed,color='C3',label="Filtered")
    plt.xlabel("Incremental Angle")
    plt.ylabel("Radius")
    plt.legend(loc="upper left")
    plt.show()

# Return Spiral Quality Metrics


d=smoothed_abs
t=angle

d = np.asarray([float(i) for i in d])
t = np.asarray([float(i) for i in t])

peak = np.asarray(find_peaks(d,distance=5)[0])


if not verbose == None:
    plt.plot(t,d)
    plt.plot(t[peak], d[peak], "*")
    plt.show()


av_peak_height = np.mean(d[peak])


lin_transform_min = 0.4906734
lin_transform_max = 0.9854416


# mean_error = (math.log(mean_error,10) + lin_transform_min)/lin_transform_max

# if mean_error < 0:
#     mean_error = 0

# mean_error = mean_error*9.5

print(mean_error,std_error,max_error,fit_error,av_peak_height,len(angle))



# Save updated input parameters

if not sc == None or not thresh == None:

    command_history = shelve.open(command_dict)

    command_history['sc'] = starting_coords
    command_history['thresh'] = thresh

    command_history.close()




