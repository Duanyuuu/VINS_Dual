#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <mutex>
#include <thread>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_img_right, pub_match_right;
ros::Publisher pub_restart;

FeatureTracker trackerData[2];
double first_image_time;
double first_image_time1;
int pub_count = 1;
int pub_count1 = 1;
bool first_image_flag = true;
bool first_image_flag1 = true;
double last_image_time = 0;
double last_image_time1 = 0;
bool init_pub = 0;
bool init_pub1 = 0;

void img_callback (const sensor_msgs::ImageConstPtr &img_msg) { 
    cout << "aaaaaaaa" << endl;
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback (const sensor_msgs::ImageConstPtr &img_msg) {
    cout << "bbbbbbbb" << endl;
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

cv_bridge::CvImageConstPtr getImgFromeMsgToCv(const sensor_msgs::ImageConstPtr &img_msg, cv_bridge::CvImageConstPtr ptr)
{
    // cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else 
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    return ptr;
}

void pubFeatures (const sensor_msgs::ImageConstPtr &ref_img_msg, const sensor_msgs::ImageConstPtr &right_img_msg) {
    sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
    sensor_msgs::ChannelFloat32 id_of_point;
    sensor_msgs::ChannelFloat32 u_of_point;
    sensor_msgs::ChannelFloat32 v_of_point;
    sensor_msgs::ChannelFloat32 velocity_x_of_point;
    sensor_msgs::ChannelFloat32 velocity_y_of_point;

    if (ref_img_msg) 
        feature_points->header = ref_img_msg->header;
    feature_points->header.frame_id = "world";

    vector<set<int>> hash_ids(NUM_OF_CAM);

    auto &un_pts = trackerData[0].cur_un_pts;
    auto &cur_pts = trackerData[0].cur_pts;
    auto &ids = trackerData[0].ids;
    auto &pts_velocity = trackerData[0].pts_velocity;
    
    for (unsigned int j = 0; j < ids.size(); j++)
    {
        if (trackerData[0].track_cnt[j] > 1)    //第j个特征点被第i个相机跟踪的次数大于1
        {
            int p_id = ids[j];
            hash_ids[0].insert(p_id);
            geometry_msgs::Point32 p;
            p.x = un_pts[j].x;
            p.y = un_pts[j].y;
            p.z = 1;

            feature_points->points.push_back(p);
            id_of_point.values.push_back(p_id);
            u_of_point.values.push_back(cur_pts[j].x);
            v_of_point.values.push_back(cur_pts[j].y);
            velocity_x_of_point.values.push_back(pts_velocity[j].x);
            velocity_y_of_point.values.push_back(pts_velocity[j].y);
        }
    }

    feature_points->channels.push_back(id_of_point);
    feature_points->channels.push_back(u_of_point);
    feature_points->channels.push_back(v_of_point);
    feature_points->channels.push_back(velocity_x_of_point);
    feature_points->channels.push_back(velocity_y_of_point);
    ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
    // skip the first image; since no optical speed on frist image
    if (!init_pub)
    {
        init_pub = 1;   //第一帧不发布，因为没有光流速度
    }
    else
        pub_img.publish(feature_points);
    
    if (right_img_msg && feature_points) {
        cout << "pub_right_features" << endl;
        un_pts = trackerData[1].cur_un_pts;
        cur_pts = trackerData[1].cur_pts;
        ids = trackerData[1].ids;
        pts_velocity = trackerData[1].pts_velocity;

        for (unsigned int j = 0; j < ids.size(); j++)
        {
            if (trackerData[1].track_cnt[j] > 1)    //第j个特征点被第i个相机跟踪的次数大于1
            {
                int p_id = ids[j];
                hash_ids[1].insert(p_id);
                geometry_msgs::Point32 p;
                p.x = un_pts[j].x;
                p.y = un_pts[j].y;
                p.z = 1;

                feature_points->points[0] = p;
                id_of_point.values[0] = p_id;
                u_of_point.values[0] = cur_pts[j].x;
                v_of_point.values[0] = cur_pts[j].y;
                velocity_x_of_point.values[0] = pts_velocity[j].x;
                velocity_y_of_point.values[0] = pts_velocity[j].y;
            }
        }

        feature_points->channels[0] = id_of_point;
        feature_points->channels[0] = u_of_point;
        feature_points->channels[0] = v_of_point;
        feature_points->channels[0] = velocity_x_of_point;
        feature_points->channels[0] = velocity_y_of_point;
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub1)
        {
            init_pub1 = 1;   //第一帧不发布，因为没有光流速度
        }
        else
            pub_img_right.publish(feature_points);
    }
}

int cnt;

void f_process() {
    while (true) {
        sensor_msgs::ImageConstPtr img0_msg;
        sensor_msgs::ImageConstPtr img1_msg;
        double time = 0;
        std_msgs::Header header;
        cv_bridge::CvImageConstPtr ptr_left;
        cv_bridge::CvImageConstPtr ptr_right;
        vector<cv::Mat> show_img;
        m_buf.lock();
        if (!img0_buf.empty()) {
            ROS_INFO("image0_buf is not empty,size is %d", img0_buf.size());
            PUB_THIS_FRAME = true;
            img0_msg = img0_buf.front();
            time = img0_buf.front()->header.stamp.toSec();
            header = img0_buf.front()->header;
            ptr_left = getImgFromeMsgToCv(img0_buf.front(), ptr_left);
            img0_buf.pop();
            if (!img1_buf.empty()) {
                ROS_INFO("image1_buf is not empty,size is %d", img1_buf.size());
                PUB_THIS_FRAME1 = true;
                img1_msg = img1_buf.front();
                ptr_right = getImgFromeMsgToCv(img1_buf.front(), ptr_right);
                img1_buf.pop();
            }
        }
        m_buf.unlock();
        if (ptr_left) {
            cout << "-------show_img " << cnt++ << "-------" << endl;
            show_img.push_back(ptr_left->image);
            if (ptr_right) {
                cout << "-------show_img " << cnt++ << "-------" << endl;
                show_img.push_back(ptr_right->image);
            }
        }
        if (use_stereo) {
            if (ptr_left && ptr_right) {
                ROS_INFO("left and right buf both have images");
                // stereoTrack(ptr_left->image, ptr_left->image, trackerData, time);
                stereoReadeImg(ptr_left->image, ptr_left->image, trackerData[0], trackerData[1], time);
            }
            
            drawTrack(trackerData[0], trackerData[1]);

            trackerData[0].prevLeftPtsMap.clear();
            for (size_t i = 0; i < trackerData[0].cur_pts.size(); i++) 
                trackerData[0].prevLeftPtsMap[trackerData[0].ids[i]] = trackerData[0].cur_pts[i];

            
            for (unsigned int i = 0;; i++) {
                bool completed = false;
                completed |= trackerData[0].updateID(i);

                if (!completed)
                    break;
            }

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(time);
            sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", trackerData[0].imTrack).toImageMsg();
            pub_match.publish(imgTrackMsg);
                       
        }
        else {
            if (ptr_left) 
                trackerData[0].readImage(ptr_left->image, time);
            
            if (ptr_right) 
                trackerData[1].readImage(ptr_left->image, time);
                
            // for (unsigned int i = 0; ; i++) {
            //     bool completed = false;
            //     completed |= trackerData[0].updateID(i);
            //     if (!completed)
            //         break;
            // }
            for (unsigned int i = 0;; i++)
            {
                bool completed = false;
                for (int j = 0; j < NUM_OF_CAM; j++)
                    if (j != 1 || !STEREO_TRACK)
                        completed |= trackerData[j].updateID(i);    //更新特征点id
                if (!completed)
                    break;
            }

            if (PUB_THIS_FRAME && PUB_THIS_FRAME1) {
                pub_count++;
                
                pubFeatures(img0_msg, img1_msg);

                // 将图像封装到cv_bridge::cvtColor类型的ptr实例中发布到pub_match;
                if (SHOW_TRACK)
                {
                    if (!ptr_left || !ptr_right) continue;
                    // cout << "ptr!!!!!!" << endl;
                    ptr_left = cv_bridge::cvtColor(ptr_left, sensor_msgs::image_encodings::BGR8);
                    ptr_right = cv_bridge::cvtColor(ptr_right, sensor_msgs::image_encodings::BGR8);
                
                    cv::Mat stereo_img;
                    cv::hconcat(ptr_left->image, ptr_right->image, stereo_img);
                    for (int i = 0; i < NUM_OF_CAM; i++) {
                        cv::Mat tmp_img = stereo_img.colRange(i * COL, (i+1) * COL);
                        cv::cvtColor(show_img[i], tmp_img, CV_GRAY2RGB);
                        //显示追踪状态，越红越好，越蓝越不行
                        for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                        {
                            double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                            cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                        }
                    }
                    std_msgs::Header header;
                    header.frame_id = "world";
                    header.stamp = ros::Time(time);
                    sensor_msgs::ImagePtr stereoImgTrack = cv_bridge::CvImage(header, "bgr8", stereo_img).toImageMsg();
                    pub_match.publish(stereoImgTrack);
                }
            }
        }
    }
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
    ros::Subscriber sub_img1 = n.subscribe(IMAGE_TOPIC1, 100, img1_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_img_right = n.advertise<sensor_msgs::PointCloud>("feature_right", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    // pub_match_right = n.advertise<sensor_msgs::Image>("feature_img_right", 1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    std::thread f_thread{f_process};
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?