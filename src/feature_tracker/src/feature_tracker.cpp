#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

void stereoReadeImg(const cv::Mat &_img, const cv::Mat &_img1, FeatureTracker &_trackerDataImg0, FeatureTracker &_trackerDataImg1, double _cur_time) {
    cv::Mat img;
    cv::Mat img1;
    TicToc t_r;
    _trackerDataImg0.cur_time = _cur_time;
    _trackerDataImg1.cur_time = _cur_time;

    
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        clahe->apply(_img1, img1);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else {
        img = _img;
        img1 = _img1;
    }

    if (_trackerDataImg0.forw_img.empty() || _trackerDataImg1.forw_img.empty()) {
        _trackerDataImg0.prev_img = _trackerDataImg0.cur_img = _trackerDataImg0.forw_img = img;
        _trackerDataImg1.prev_img = _trackerDataImg1.cur_img = _trackerDataImg1.forw_img = img1;
    }
    else {
        _trackerDataImg0.forw_img = img;
        _trackerDataImg1.forw_img = img1;
    }

    _trackerDataImg0.forw_pts.clear();
    _trackerDataImg1.forw_pts.clear();
    // _trackerDataImg1.ids.clear();
    // cout << "size of img0's track_cnt is: " << _trackerDataImg0.track_cnt.size() << endl;
    // cout << "size of img1's track_cnt is: " << _trackerDataImg1.track_cnt.size() << endl;
    if (_trackerDataImg0.cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status, status1;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(_trackerDataImg0.cur_img, _trackerDataImg0.forw_img, _trackerDataImg0.cur_pts, _trackerDataImg0.forw_pts, status, err, cv::Size(21, 21), 3);

        cout << "status size: " << status.size() << endl;
        int n = 0;
        for (int i = 0; i < int(status.size()); i++) {
            if (status[i]) {
                n++;
            }
        }
        cout << "status is 1: " << n << endl;

        for (int i = 0; i < int(_trackerDataImg0.forw_pts.size()); i++)
            if (status[i] && !inBorder(_trackerDataImg0.forw_pts[i]))
                status[i] = 0;
        reduceVector(_trackerDataImg0.prev_pts, status);
        reduceVector(_trackerDataImg0.cur_pts, status);
        reduceVector(_trackerDataImg0.forw_pts, status);
        reduceVector(_trackerDataImg0.ids, status);
        reduceVector(_trackerDataImg0.cur_un_pts, status);
        reduceVector(_trackerDataImg0.track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());

        cout << "1 size of img0's forw_pts is: " << _trackerDataImg0.forw_pts.size() << endl;
        cv::calcOpticalFlowPyrLK(_trackerDataImg0.forw_img, _trackerDataImg1.forw_img, _trackerDataImg0.forw_pts, _trackerDataImg1.forw_pts, status1, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(_trackerDataImg0.forw_pts.size()); i++)
            if (status[i] && !inBorder(_trackerDataImg1.forw_pts[i]) && status1[i])
                status[i] = 0;
        
        reduceVector(_trackerDataImg0.cur_pts, status);
        reduceVector(_trackerDataImg1.cur_pts, status);
        reduceVector(_trackerDataImg0.forw_pts, status);
        reduceVector(_trackerDataImg1.forw_pts, status);
        reduceVector(_trackerDataImg0.ids, status);
        reduceVector(_trackerDataImg1.ids, status);
        reduceVector(_trackerDataImg0.cur_un_pts, status);
        reduceVector(_trackerDataImg1.cur_un_pts, status);
        reduceVector(_trackerDataImg0.track_cnt, status);
        reduceVector(_trackerDataImg1.track_cnt, status);

        cout << "3 size of img0's forw_pts is: " << _trackerDataImg0.forw_pts.size() << endl;
        // cout << "size of img0's track_cnt is: " << _trackerDataImg0.track_cnt.size() << endl;
        // cout << "size of img1's track_cnt is: " << _trackerDataImg1.track_cnt.size() << endl;
    }

    for (auto &n : _trackerDataImg0.track_cnt)
        n++;

    if (PUB_THIS_FRAME && PUB_THIS_FRAME1) {
        _trackerDataImg0.rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        _trackerDataImg0.setMask();
        // if (FISHEYE) {
        //     _trackerDataImg0.mask = _trackerDataImg0.fisheye_mask.clone();
        //     _trackerDataImg1.mask = _trackerDataImg1.fisheye_mask.clone();
        // }
        // else {
        //     _trackerDataImg0.mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
        //     _trackerDataImg1.mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
        // }

        // vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
        // vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id_right;

        // for (unsigned int i = 0; i < _trackerDataImg0.forw_pts.size(); i++)
        //     cnt_pts_id.push_back(make_pair(_trackerDataImg0.track_cnt[i], make_pair(_trackerDataImg0.forw_pts[i], _trackerDataImg0.ids[i])));

        // sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
        //     {
        //         return a.first > b.first;
        //     });
        
        // cout << "debug2" << endl;
        // cout << "size of img1's forw_pts is: " << _trackerDataImg1.forw_pts.size() << endl;
        // cout << "size of img1's track_cnt is: " << _trackerDataImg1.track_cnt.size() << endl;
        // cout << "size of img1's ids is: " << _trackerDataImg1.ids.size() << endl;
        // for (unsigned int i = 0; i < _trackerDataImg1.forw_pts.size(); i++) {
        //     cnt_pts_id_right.push_back(make_pair(_trackerDataImg0.track_cnt[i], make_pair(_trackerDataImg1.forw_pts[i], _trackerDataImg1.ids[i])));
        // }

        // cout << "debug3" << endl;
        // sort(cnt_pts_id_right.begin(), cnt_pts_id_right.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
        //     {
        //         return a.first > b.first;
        //     });

        // _trackerDataImg0.forw_pts.clear();
        // _trackerDataImg1.forw_pts.clear();
        // _trackerDataImg0.ids.clear();
        // _trackerDataImg1.ids.clear();
        // _trackerDataImg0.track_cnt.clear();
        // _trackerDataImg1.track_cnt.clear();

        // auto it_right = cnt_pts_id_right.begin();
        // for (auto &it : cnt_pts_id)
        // {
        //     if (_trackerDataImg0.mask.at<uchar>(it.second.first) == 255)// this place no features
        //     {
        //         _trackerDataImg0.forw_pts.push_back(it.second.first);
        //         _trackerDataImg1.forw_pts.push_back((*it_right).second.first);
        //         _trackerDataImg0.ids.push_back(it.second.second);
        //         _trackerDataImg0.track_cnt.push_back(it.first);
        //         cv::circle(_trackerDataImg0.mask, it.second.first, MIN_DIST, 0, -1);
        //     }
        //     it_right++;
        // }
        // _trackerDataImg1.ids = _trackerDataImg0.ids;
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        // detect new features
        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(_trackerDataImg0.forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(_trackerDataImg0.mask.empty())
                cout << "mask is empty " << endl;
            if (_trackerDataImg0.mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (_trackerDataImg0.mask.size() != _trackerDataImg0.forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(_trackerDataImg0.forw_img, _trackerDataImg0.n_pts, MAX_CNT - _trackerDataImg0.forw_pts.size(), 0.01, MIN_DIST, _trackerDataImg0.mask);
        }
        else
            _trackerDataImg0.n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        _trackerDataImg0.addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    _trackerDataImg0.prev_img = _trackerDataImg0.cur_img;
    _trackerDataImg0.prev_pts = _trackerDataImg0.cur_pts;
    _trackerDataImg0.prev_un_pts = _trackerDataImg0.cur_un_pts;
    _trackerDataImg0.cur_img = _trackerDataImg0.forw_img;
    _trackerDataImg1.cur_img = _trackerDataImg1.forw_img;
    _trackerDataImg0.cur_pts = _trackerDataImg0.forw_pts;
    _trackerDataImg1.cur_pts = _trackerDataImg1.forw_pts;
    _trackerDataImg0.undistortedPoints();
    _trackerDataImg0.prev_time = _trackerDataImg0.cur_time;
    _trackerDataImg1.prev_time = _trackerDataImg0.cur_time;

    cout << "right cur_pts: " << _trackerDataImg1.cur_pts.size() << endl;
    cout << "right forw_pts: " << _trackerDataImg1.forw_pts.size() << endl;
}

void stereoTrack(const cv::Mat &_img, const cv::Mat &_img1, FeatureTracker *_trackerData, double _cur_time)
{
    cv::Mat img;
    cv::Mat img1;
    TicToc t_r;
    _trackerData[0].cur_img = _cur_time;
    _trackerData[1].cur_img = _cur_time;

    if (EQUALIZE) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(_img, img);
        clahe->apply(_img1, img1);
    }
    else {
        img = _img;
        img1 = _img1;
    }

    if (_trackerData[0].forw_img.empty() || _trackerData[1].forw_img.empty()) {
        _trackerData[0].cur_img = _trackerData[0].forw_img = img;
        _trackerData[1].cur_img = _trackerData[1].forw_img = img1;
    }
    else {
        _trackerData[0].forw_img = img;
        _trackerData[1].forw_img = img1;
    }

    _trackerData[0].forw_pts.clear();
    _trackerData[1].forw_pts.clear();
    _trackerData[1].ids.clear();

    // stereo flow track
    // cout << "cur_pts1: " << _trackerData[0].cur_pts.size() << endl;
    if (_trackerData[0].cur_pts.size() > 0) {
        vector<uchar> status,status1;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(_trackerData[0].cur_img, _trackerData[0].forw_img, 
                                    _trackerData[0].cur_pts, _trackerData[0].forw_pts, 
                                    status, err, cv::Size(21, 21), 3);
        cout << "status size: " << status.size() << endl;
        int n = 0;
        for (int i = 0; i < int(status.size()); i++) {
            if (status[i] == 1) {
                n++;
            }
        }
        cout << "status is 1: " << n << endl;
        for (int i = 0; i < int(_trackerData[0].forw_pts.size()); i++) {
            if (status[i] && !inBorder(_trackerData[0].forw_pts[i])) 
                status[i] = 0;
        }

        // cout << "status is 1: " << n << endl;
        reduceVector(_trackerData[0].cur_pts, status);
        // cout << "cur_pts2: " << _trackerData[0].cur_pts.size() << endl;
        reduceVector(_trackerData[0].forw_pts, status);
        reduceVector(_trackerData[0].ids, status);
        reduceVector(_trackerData[0].cur_un_pts,status);
        reduceVector(_trackerData[0].track_cnt,status);
        
        cv::calcOpticalFlowPyrLK(_trackerData[0].forw_img, _trackerData[1].forw_img, 
                                    _trackerData[0].forw_pts, _trackerData[1].forw_pts, 
                                    status1, err, cv::Size(21, 21), 3);
        // for (int i = 0; i < int(_trackerData[0].forw_pts.size()); i++) {
        //     if (status[i] && !inBorder(_trackerData[1].forw_pts[i]) && status1[i])
        //         status[i] = 0;
        // }
        // reduceVector(_trackerData[0].cur_pts, status);
        // cout << "cur_pts3: " << _trackerData[0].cur_pts.size() << endl;
        // reduceVector(_trackerData[1].cur_pts, status);
        // reduceVector(_trackerData[0].forw_pts, status);
        // reduceVector(_trackerData[1].forw_pts, status);
        // reduceVector(_trackerData[0].ids, status);
        // reduceVector(_trackerData[1].ids, status);
        // // reduceVector(_trackerDate[0].cur_un_pts,status);
        // reduceVector(_trackerData[0].track_cnt, status);
        // reduceVector(_trackerData[1].track_cnt, status);
    }

    // update track num
    for (auto &n : _trackerData[0].track_cnt) 
        n++;

    // track new features
    if (PUB_THIS_FRAME && PUB_THIS_FRAME1) {

        // _trackerData[0].setMask();
        // setMask
        if (FISHEYE) {
            _trackerData[0].mask = _trackerData[0].fisheye_mask.clone();
            _trackerData[1].mask = _trackerData[1].fisheye_mask.clone();
        }
        else {
            _trackerData[0].mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
            _trackerData[1].mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
        }

        vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
        vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id_right;

        for (unsigned int i = 0; i < _trackerData[0].forw_pts.size(); i++)
            cnt_pts_id.push_back(make_pair(_trackerData[0].track_cnt[i], make_pair(_trackerData[0].forw_pts[i], _trackerData[0].ids[i])));

        sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
            {
                return a.first > b.first;
            });

        for (unsigned int i = 0; i < _trackerData[1].forw_pts.size(); i++)
            cnt_pts_id_right.push_back(make_pair(_trackerData[0].track_cnt[i], make_pair(_trackerData[1].forw_pts[i], _trackerData[0].ids[i])));

        sort(cnt_pts_id_right.begin(), cnt_pts_id_right.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
            {
                return a.first > b.first;
            });

        _trackerData[0].forw_pts.clear();
        _trackerData[1].forw_pts.clear();
        _trackerData[0].ids.clear();
        _trackerData[1].ids.clear();
        _trackerData[0].track_cnt.clear();

        auto it_right = cnt_pts_id_right.begin();
        for (auto &it : cnt_pts_id)
        {
            if (_trackerData[0].mask.at<uchar>(it.second.first) == 255)// this place no features
            {
                _trackerData[0].forw_pts.push_back(it.second.first);
                _trackerData[1].forw_pts.push_back((*it_right).second.first);
                _trackerData[0].ids.push_back(it.second.second);
                _trackerData[0].track_cnt.push_back(it.first);
                cv::circle(_trackerData[0].mask, it.second.first, MIN_DIST, 0, -1);
            }
            it_right++;
        }
        _trackerData[1].ids = _trackerData[0].ids;
        
        // detect new features
        int n_max_cnt = MAX_CNT - static_cast<int>(_trackerData[0].forw_pts.size());
        if (n_max_cnt > 0) {
            if (_trackerData[0].mask.empty()) 
                cout << "mask is empty " << endl;
            if (_trackerData[0].mask.type() != CV_8UC1) 
                cout << "mask type wrong" << endl;
            if (_trackerData[0].mask.size() != _trackerData[0].forw_img.size())
                cout << "wrong size" << endl;
            cv::goodFeaturesToTrack(_trackerData[0].forw_img, _trackerData[0].n_pts, MAX_CNT - _trackerData[0].forw_pts.size(), 0.01, MIN_DIST, _trackerData[0].mask);        
        }
        else 
            _trackerData[0].n_pts.clear();
        
        // add new features
        // 这里只需要增加左目特征点
        _trackerData[0].addPoints();
    }

    

    _trackerData[0].cur_img = _trackerData[0].forw_img;
    _trackerData[1].cur_img = _trackerData[1].forw_img;

    _trackerData[0].cur_pts.clear();
    _trackerData[1].cur_pts.clear();
    _trackerData[0].cur_pts = _trackerData[0].forw_pts;     // debug
    cout << "cur_pts4: " << _trackerData[0].cur_pts.size() << endl;
    _trackerData[1].cur_pts = _trackerData[1].forw_pts;

    // // undistortedPoints
    // _trackerData[0].cur_un_pts.clear();
    // _trackerData[1].cur_un_pts.clear();
    // _trackerData[0].cur_un_pts_map.clear();

    // for (unsigned int i = 0; i < _trackerData[0].cur_pts.size(); i++) {
    //     Eigen::Vector2d a(_trackerData[0].cur_pts[i].x, _trackerData[0].cur_pts[i].y);
    //     Eigen::Vector3d b;
    //     _trackerData[0].m_camera->liftProjective(a, b);
    //     _trackerData[0].cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    //     _trackerData[0].cur_un_pts_map.insert(make_pair(_trackerData[0].ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
    // }

    // if (_trackerData[1].cur_pts.size() > 0) {
    //     for (unsigned int i = 0; i < _trackerData[1].cur_pts.size(); i++)
    //     {
    //         Eigen::Vector2d a1(_trackerData[1].cur_pts[i].x, _trackerData[1].cur_pts[i].y);
    //         Eigen::Vector3d b1;
    //         _trackerData[1].m_camera->liftProjective(a1, b1);
    //         _trackerData[1].cur_un_pts.push_back(cv::Point2f(b1.x() / b1.z(), b1.y() / b1.z()));
    //     }
    // }

    // // caculate points velocity
    // // 此处也只需要计算左目速度
    // if (!_trackerData[0].prev_un_pts_map.empty())
    // {
    //     double dt = _trackerData[0].cur_time - _trackerData[0].prev_time;
    //     _trackerData[0].pts_velocity.clear();
    //     for (unsigned int i = 0; i < _trackerData[0].cur_un_pts.size(); i++)
    //     {
    //         if (_trackerData[0].ids[i] != -1)
    //         {
    //             std::map<int, cv::Point2f>::iterator it;
    //             it = _trackerData[0].prev_un_pts_map.find(_trackerData[0].ids[i]);
    //             if (it != _trackerData[0].prev_un_pts_map.end())
    //             {
    //                 double v_x = (_trackerData[0].cur_un_pts[i].x - it->second.x) / dt;
    //                 double v_y = (_trackerData[0].cur_un_pts[i].y - it->second.y) / dt;
    //                 _trackerData[0].pts_velocity.push_back(cv::Point2f(v_x, v_y));
    //             }
    //             else
    //                 _trackerData[0].pts_velocity.push_back(cv::Point2f(0, 0));
    //         }
    //         else
    //         {
    //             _trackerData[0].pts_velocity.push_back(cv::Point2f(0, 0));
    //         }
    //     }
    // }
    // else
    // {
    //     for (unsigned int i = 0; i < _trackerData[0].cur_pts.size(); i++)
    //     {
    //         _trackerData[0].pts_velocity.push_back(cv::Point2f(0, 0));
    //     }
    // }
    // _trackerData[0].prev_un_pts_map = _trackerData[0].cur_un_pts_map;

    // // drawTrack

    // update time
    _trackerData[0].prev_time = _trackerData[0].cur_time;
    _trackerData[1].prev_time = _trackerData[0].cur_time;
}

void drawTrack(FeatureTracker &_trackerDataImg0, FeatureTracker &_trackerDataImg1) {
    //int rows = imLeft.rows;
    int cols = _trackerDataImg0.cur_img.cols;
    if (!_trackerDataImg1.cur_img.empty()) {
        cout << "!!!draw two imgs!!!" << endl;
        cv::hconcat(_trackerDataImg0.cur_img, _trackerDataImg1.cur_img, _trackerDataImg0.imTrack);
    }
    else
        _trackerDataImg0.imTrack = _trackerDataImg0.cur_img.clone();
    if (!_trackerDataImg0.imTrack.empty())
        cv::cvtColor(_trackerDataImg0.imTrack, _trackerDataImg0.imTrack, CV_GRAY2RGB);

    for (size_t j = 0; j < _trackerDataImg0.cur_pts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * _trackerDataImg0.track_cnt[j] / 20);
        cv::circle(_trackerDataImg0.imTrack, _trackerDataImg0.cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!_trackerDataImg1.cur_img.empty())
    {
        cout << "img1_pts: " << _trackerDataImg1.cur_pts.size() << endl;
        for (size_t i = 0; i < _trackerDataImg1.cur_pts.size(); i++)
        {
            cv::Point2f rightPt = _trackerDataImg1.cur_pts[i];
            rightPt.x += cols;
            cv::circle(_trackerDataImg0.imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
        }
    }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < _trackerDataImg0.ids.size(); i++)
    {
        int id = _trackerDataImg0.ids[i];
        mapIt = _trackerDataImg0.prevLeftPtsMap.find(id);
        if(mapIt != _trackerDataImg0.prevLeftPtsMap.end())
            cv::arrowedLine(_trackerDataImg0.imTrack, _trackerDataImg0.cur_pts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
