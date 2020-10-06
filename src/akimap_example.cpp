#include <ros/ros.h>
#include <akimap/akimap.h>

#include <tf/transform_listener.h>
#include <tf/message_filter.h>

#include <message_filters/subscriber.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

class OnTheFlyMapping {
public:
    OnTheFlyMapping() {
        nh.getParam("fixed_frame_id", FIXED_FRAME_ID);
        pc_subscriber = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/pointcloud_in", 1);
        tf_pc_subscriber = new tf::MessageFilter<sensor_msgs::PointCloud2>(*pc_subscriber, tf_listener, FIXED_FRAME_ID, 1);
        tf_pc_subscriber->registerCallback(boost::bind(&OnTheFlyMapping::update_occupancy_map, this, _1));

        occupied_cells_publisher = nh.advertise<sensor_msgs::PointCloud2>("/akimap/occupied_cells", 1);

        double resolution, free_sample_distance;
        nh.getParam("resolution", resolution);
        nh.getParam("free_sample_distance", free_sample_distance);

        map = new AKIMap(resolution, free_sample_distance);
    }
    ~OnTheFlyMapping() {
        delete map;
    }

    void update_occupancy_map(const sensor_msgs::PointCloud2ConstPtr& _msg_pointcloud) {
        if(map == nullptr)
            return;

        // 1. Parse a point cloud and its origin
        Eigen::MatrixXd pointcloud;
        Eigen::Vector3d origin;
        {
            tf::StampedTransform sensor_to_world_tf;
            try {
                tf_listener.lookupTransform(FIXED_FRAME_ID, _msg_pointcloud->header.frame_id, _msg_pointcloud->header.stamp, sensor_to_world_tf);
            }
            catch(tf::TransformException& e) {
                ROS_ERROR_STREAM("Transform error of sensor data: " << e.what() << ", quitting callback");
                return;
            }

            Eigen::Matrix4f sensor_to_world;
            pcl_ros::transformAsMatrix(sensor_to_world_tf, sensor_to_world);

            pcl::PointCloud<pcl::PointXYZ> pcl_pointcloud;
            pcl::fromROSMsg(*_msg_pointcloud, pcl_pointcloud);
            pcl::transformPointCloud(pcl_pointcloud, pcl_pointcloud, sensor_to_world);

            pointcloud.resize(3, pcl_pointcloud.size());
            for(int i = 0; i < pcl_pointcloud.size(); i++)
                pointcloud.col(i) << pcl_pointcloud[i].x, pcl_pointcloud[i].y, pcl_pointcloud[i].z;
            origin << sensor_to_world_tf.getOrigin().x(), sensor_to_world_tf.getOrigin().y(), sensor_to_world_tf.getOrigin().z();
        }

        // 2. Update the occupancy map
        ros::WallTime update_start = ros::WallTime::now();
        map->insert_pointcloud(pointcloud, origin);
        ros::WallTime update_end = ros::WallTime::now();
        double update_time_elapsed_in_scan = (update_end - update_start).toSec() * 1000;  // [ms]
        ROS_INFO("update time: %lf [ms]", update_time_elapsed_in_scan);

        // 3. Visualize the occupied cells
        sensor_msgs::PointCloud2 msg_pointcloud;
        {
            pcl::PointCloud<pcl::PointXYZ> pcl_pointcloud;
            for(const auto& cell : map->get_nodes()) {
                if(cell.second->get_occupancy() > OCCUPIED_THRESHOLD) {
                    pcl::PointXYZ data;
                    data.x = (float)map->key_to_coordinate(cell.first[0]);
                    data.y = (float)map->key_to_coordinate(cell.first[1]);
                    data.z = (float)map->key_to_coordinate(cell.first[2]);
                    pcl_pointcloud.push_back(data);
                }
            }

            pcl::toROSMsg(pcl_pointcloud, msg_pointcloud);
            msg_pointcloud.header.frame_id = FIXED_FRAME_ID;
            msg_pointcloud.header.stamp = ros::Time::now();
        }
        occupied_cells_publisher.publish(msg_pointcloud);
    }

protected:
    AKIMap* map;

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::PointCloud2>* pc_subscriber;
    tf::MessageFilter<sensor_msgs::PointCloud2>* tf_pc_subscriber;
    tf::TransformListener tf_listener;

    ros::Publisher occupied_cells_publisher;
    const double OCCUPIED_THRESHOLD = 0.5;

    std::string FIXED_FRAME_ID;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "akimap_example");
    ros::NodeHandle nh;

    OnTheFlyMapping mapping_server;

    try {
        ros::spin();
    }
    catch(std::runtime_error& e) {
        ROS_ERROR("Exception: %s", e.what());
        return -1;
    }

    return 0;
}