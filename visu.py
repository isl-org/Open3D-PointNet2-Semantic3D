import argparse
import numpy as np
import os
import importlib
import json
import utils.pc_util as pc_util

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--set', default="train", help='train or test [default: train]')
parser.add_argument('--n', type=int, default=8, help='Number of batches you want [default : 1]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size [default: 32]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 8192]')
parser.add_argument('--dataset', default='semantic', help='Dataset [default: semantic_color]')
parser.add_argument('--config', type=str, default="config.json", metavar='N', help='config file')
FLAGS = parser.parse_args()

JSON_DATA_CUSTOM = open(FLAGS.config).read()
CUSTOM = json.loads(JSON_DATA_CUSTOM)
JSON_DATA = open('default.json').read()
PARAMS = json.loads(JSON_DATA)
PARAMS.update(CUSTOM)

SET = FLAGS.set
NBATCH = FLAGS.n
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATASET_NAME = FLAGS.dataset

DROPOUT = False
DROPOUT_RATIO = 0.875
AUGMENT = False
STATS = True
DRAW_PLOTS = True
STAT_INPUT_NB = 1000 # per scene of the dataset
MAX_EXPORT = 20 # the maximum number of scenes to be exported
GROUP_BY_BATCHES = True


if DROPOUT:
    print("dropout is on, with ratio %f" %(DROPOUT_RATIO))
if AUGMENT:
    print ("rotation is on")

# Import dataset
data = importlib.import_module('dataset.' + DATASET_NAME)
DATA = data.Dataset(npoints=NUM_POINT, split=SET, box_size=PARAMS['box_size'], use_color=PARAMS['use_color'],
                             proba_terrain=PARAMS['proba_terrain'], dropout_max=PARAMS['input_dropout'], path=PARAMS['data_path'], accept_rate=PARAMS['accept_rate'])
NUM_CLASSES = DATA.num_classes
# Outputs

OUTPUT_DIR = os.path.join("visu", DATASET_NAME+"_"+SET)
if not os.path.exists("visu"): os.mkdir("visu")
if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
        
OUTPUT_DIR_HIST    = os.path.join(OUTPUT_DIR, "hist")
OUTPUT_DIR_STATS   = os.path.join(OUTPUT_DIR, "color_proba_selection")
OUTPUT_DIR_SEEDS   = os.path.join(OUTPUT_DIR, "seeds")
OUTPUT_DIR_BATCHES = os.path.join(OUTPUT_DIR, "grouped_by_batches")
    
if STATS:
    """
    Here we want to get info about class representation in decimated data 
    and in the data we feed to the network.
    We also want to know whether the data is well fed by the sampling algorithm
    because we don't want to focus too much on always the same points. Thus we
    compute an estimator of the probability of being selected when a scene is
    considered for input points, and we export of map of probability 
    (one per scene).
    Moreover we export with each point cloud a cloud containing the seeds. You
    may visualise with CloudCompare by loading it separately from the cloud and
    setting a big point size for the seeds.
    """
    if not os.path.exists(OUTPUT_DIR_HIST): os.mkdir(OUTPUT_DIR_HIST)
    if not os.path.exists(OUTPUT_DIR_STATS): os.mkdir(OUTPUT_DIR_STATS)
    if not os.path.exists(OUTPUT_DIR_SEEDS): os.mkdir(OUTPUT_DIR_SEEDS)
    import matplotlib.pyplot as plt
    
    # the histogram of the decimated point cloud labels
    if DRAW_PLOTS:
        label_hist = DATA.get_hist()
        label_hist = np.array(label_hist)
        density_hist = label_hist/np.sum(label_hist)
        plt.bar(range(NUM_CLASSES), density_hist, color='green')
        #plt.yscale('log')
        plt.xlabel('Classes : {}'.format(DATA.get_list_classes_str()))
        plt.xticks(range(NUM_CLASSES))
        plt.ylabel('Proportion')
        plt.title('Histogram of class repartition in decimated point clouds of set {}'.format(SET))
        plt.draw()
        plt.savefig(os.path.join(OUTPUT_DIR_HIST,"dec.png"))
    
    # generate real inputs as fed into the network
    filenames = DATA.get_data_filenames()
    print("Generating {} inputs".format(STAT_INPUT_NB*len(filenames)))
    pc_shapes=list()
    hist_list = list() # counts for every point of every scene the nb of times it is taken as input
    seeds_list=list()
    label_hist = np.zeros(NUM_CLASSES)
    scene_counters = np.zeros(len(filenames))
    for f in range(len(filenames)):
        pc_shapes.append(DATA.get_scene_shape(f))
        hist_list.append(np.zeros(pc_shapes[f][0]))
        seeds_list.append(list())
    #if len(np.unique(np.array(pc_shapes))) < len(pc_shapes):
        #print("WARNING : scenes cannot be distinguished by shape; data generated into "+OUTPUT_DIR_STATS+" and "+OUTPUT_DIR_SEEDS+" may be erroneous")
    for i in range(STAT_INPUT_NB*len(filenames)):
        if i%100==0 and i>0:
            print("{} inputs generated".format(i))
        f, input_mask, input_label_hist, seed_idx = DATA.next_input(DROPOUT, True, False, True)
        hist_list[f]+=input_mask
        scene_counters[f]+=1
        seeds_list[f].append(seed_idx)
        label_hist+=input_label_hist
    
    # the histogram of the input point cloud labels
    label_hist = np.array(label_hist)
    density_hist = label_hist/np.sum(label_hist)
    plt.figure(2)
    plt.bar(range(NUM_CLASSES), density_hist, color='green')
    #plt.yscale('log')
    plt.xlabel('Classes : {}'.format(DATA.get_list_classes_str()))
    plt.xticks(range(NUM_CLASSES))
    plt.ylabel('Proportion')
    plt.title('Histogram of class repartition in input point clouds of set {}'.format(SET))
    plt.draw()        
    plt.savefig(os.path.join(OUTPUT_DIR_HIST,"input.png"))

    # the histogram of the probability of being fed into the network
    filenamesForExport = filenames[0:min(len(filenames), MAX_EXPORT)]
    for f, filename in enumerate(filenamesForExport):
        if scene_counters[f]==0:
            continue
        if len(filenames) < 4:
            plt.figure(3+f)
        else:
            plt.figure(3)
        density = 100*hist_list[f]/scene_counters[f]
        plt.hist(density, bins=20, color='green')
        plt.yscale('log')
        plt.xlabel(r'Probability of occurence in %10^{-2}')
        plt.ylabel('Proportion')
        plt.title('Histogram of selection likelihood in scene {} of set {}'.format(filename, SET))
        plt.draw()
        plt.savefig(os.path.join(OUTPUT_DIR_HIST,"proba_scene_"+str(f)+".png"))
    if DRAW_PLOTS and len(filenames) < 4:
        plt.show()
    
    # exporting the probability of being selected and the point seeds.
    print("exporting {} point clouds with densities".format(len(filenamesForExport)))
    for f, filename in enumerate(filenamesForExport):
        if np.sum(hist_list[f])==0:
            continue
        density = hist_list[f]/np.sum(hist_list[f])
        point_cloud = DATA.get_scene(f)
        # all points 
        np.savetxt(os.path.join(OUTPUT_DIR_STATS,"{}_{}_{}.txt".format(SET, os.path.basename(filename), "appearance_density")), np.hstack((point_cloud, density.reshape((-1,1)))), delimiter=" ")
        # this time with a mask removing points that weren't seen at all
        mask = density > 0
        np.savetxt(os.path.join(OUTPUT_DIR_STATS,"{}_{}_{}.txt".format(SET, os.path.basename(filename), "appearance_density_cleaned")), np.hstack((point_cloud, density.reshape((-1,1))))[mask], delimiter=" ")
        # now the seeds
        np.savetxt(os.path.join(OUTPUT_DIR_SEEDS,"{}_{}_{}.txt".format(SET, os.path.basename(filename), "seeds")), np.array(seeds_list[f]), delimiter=" ")

if NBATCH > 0:
    if not os.path.exists(OUTPUT_DIR_BATCHES): os.mkdir(OUTPUT_DIR_BATCHES)
    print("batch visualisation :")
if GROUP_BY_BATCHES and NBATCH > 0:
    """
    Here we export batches as they are fed into the network
    The idea is to position the scenes of the different batches on a grid
    where batches align themselves on the same x coordinates
    """
    data, _, _ = DATA.next_batch(BATCH_SIZE, AUGMENT, DROPOUT)
    data = data[0,:,0:3].reshape((-1,3))
    # get spatial dimension of input
    xmin, ymin, _ = np.min(data, axis=0)
    xmax, ymax, _ = np.max(data, axis=0)
    xsize = xmax - xmin + 5
    ysize = ymax - ymin + 5
    # initialize cloud and labels
    visu_cloud = np.array([0,0,0,0,0,0])
    visu_labels = np.array([0])
    
for i in range(NBATCH):
    data, label_data, _ = DATA.next_batch(BATCH_SIZE, AUGMENT, DROPOUT)
    print ("Processing batch number " + str(i))
        
    if GROUP_BY_BATCHES:
        positioned_data = list()
        for j, scene in enumerate(data):
            x = j*xsize
            y = i * ysize
            z = 0
            positioned_data.append(scene + np.array([x,y,z,0,0,0]) - np.min(scene, axis=0)*np.array([1,1,1,0,0,0]))
        batch_points = np.vstack(positioned_data)
        visu_cloud = np.vstack((visu_cloud, batch_points))
        batch_labels = np.hstack(label_data)
        visu_labels = np.hstack((visu_labels, batch_labels))
        
    else:
        for j, scene in enumerate(data):
            labels = label_data[j]
            #np.savetxt(OUTPUT_DIR+"/{}_{}_{}.obj".format(SET, "trueColors", j), scene, delimiter=" ")
            pc_util.write_ply_true_color(scene[:,0:3], scene[:,3:6], OUTPUT_DIR_BATCHES+"/{}_{}_{}.txt".format(SET, "trueColors", j))
            pc_util.write_ply_color(scene[:,0:3], labels, OUTPUT_DIR_BATCHES+"/{}_{}_{}.txt".format(SET, "labelColors", j), NUM_CLASSES)
            
if GROUP_BY_BATCHES and NBATCH > 0:
    pc_util.write_ply_true_color(visu_cloud[:,0:3], visu_cloud[:,3:6], OUTPUT_DIR_BATCHES+"/{}_{}.txt".format(SET, "trueColors"))
    pc_util.write_ply_color(visu_cloud[:,0:3], visu_labels, OUTPUT_DIR_BATCHES+"/{}_{}.txt".format(SET, "labelColors"), NUM_CLASSES)
print("done")
