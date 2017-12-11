#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.02
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()


    # TODO: PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.75
    axis_max = 1.3
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    # TODO: RANSAC Plane Segmentation
    plane_seg = cloud_filtered.make_segmenter()
    plane_seg.set_model_type(pcl.SACMODEL_PLANE)
    plane_seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.02
    plane_seg.set_distance_threshold(max_distance)
    inliers, coefficients = plane_seg.segment()

    # TODO: Extract inliers and outliers
    cloud_table   = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cloud_objects = pcl_to_ros(cloud_objects)

    # TODO: Euclidean Clustering
    #Euclidean clustering requires points with only spatial info.
    #Convert XYZRGB PC to XYZ then Construct K-d tree
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ##Euclidean Clustering 
    # Create a cluster extraction object
    cluster_ex = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    cluster_ex.set_ClusterTolerance(0.05)
    cluster_ex.set_MinClusterSize(10)
    cluster_ex.set_MaxClusterSize(1000)
    # Search the k-d tree for clusters
    cluster_ex.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    #cluster_indices now contains a list of indices for each cluster (a list of lists)
    cluster_indices = cluster_ex.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs: 
    # Classify the clusters!
    # empty lists to receive labels and object point clouds.
    detected_objects_labels = []
    detected_objects = []
  
    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):


        # Grab the points for the cluster

        pcl_cluster = cloud_objects.extract(pts_list)
        pcl_cluster = pcl_to_ros(pcl_cluster)

         
        # Compute the associated feature vector
        chists = compute_color_histograms(pcl_cluster, using_hsv=True)
        normals = get_normals(pcl_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        
        #feature = feature / np.sum(feature)
      
        feature =np.array(feature).reshape(1,-1) 
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)

if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering',anonymous=True)


    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud",pc2.PointCloud2, pcl_callback, queue_size=1)


    # TODO: Create Publishers
    object_markers_pub   = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects"  , DetectedObjectsArray, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub   = rospy.Publisher("/pcl_table"  , PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    


    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
