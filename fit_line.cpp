//
//  main.cpp
//  Ransac
//
//  Created by JiangZhigang on 2020/11/29.
//  Copyright © 2020 JiangZhigang. All rights reserved.
//

#include <iostream>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>

#include <random>
#include <set>


std::vector<cv::Point2f> generate_data(float k, float b, int n, float line_noise_level = 10, float random_noise_count = 200, bool add_ellipse = false){
    std::default_random_engine e(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<> u(-line_noise_level, line_noise_level);
    
    std::vector<cv::Point2f> points;
    for(int i=-n; i<n; i++){
        points.push_back(cv::Point2f(i, k*i + b + u(e)));
        
        if(!add_ellipse) continue;
        float a_ = 300, b_ = 200;
        float q = 1 - i * i / (a_ * a_);
        if(q > 0){
            float y = sqrt(b_*b_ * q);
            float r = atan(k);
            float c = cos(r);
            float s = sin(r);

            auto p1 = cv::Point2f(i, y);
            p1 = cv::Point2f(p1.x * c + p1.y * (-s), p1.x * s + p1.y * c + b);
            points.push_back(p1);
            auto p2 = cv::Point2f(i, -y);
            p2 = cv::Point2f(p2.x * c + p2.y * (-s), p2.x * s + p2.y * c + b);
            points.push_back(p2);
        }
    }
    
    std::uniform_int_distribution<> u_(-n, n);
    for(int i=0; i<random_noise_count; i++){
        points.push_back(cv::Point2f(u_(e), u_(e)));
    }
    std::shuffle(points.begin(), points.end(), e);
    return points;
}

void draw_data(cv::Mat& board, const std::vector<cv::Point2f>& points){
    int h = board.cols, w =  board.rows;
    for(auto const& point : points){
        cv::Point2f p = cv::Point2f(h/2+point.x, w/2-point.y);//origin change center
        cv::circle(board, p, 1, cv::Scalar(255, 255, 255), -1);
    }
}

void draw_line(cv::Mat& board, float k, float b, cv::Scalar color){
    int h = board.cols, w =  board.rows;
    cv::Point2f p1 = cv::Point2f(-h, -w * k + b);
    p1 = cv::Point2f(h/2+p1.x, w/2-p1.y);
    cv::Point2f p2 = cv::Point2f(h, w * k + b);
    p2 = cv::Point2f(h/2+p2.x, w/2-p2.y);
    cv::line(board, p1, p2, color, 1);
}


float calc_error(const cv::Point2f& point, float k, float b){
    // distance of point to line
    float A = k, B = -1, C = b;
    return abs(A * point.x + B * point.y + C) / sqrt(A * A + B * B);
}

float calc_error_y2(const cv::Point2f& point, float k, float b){
    return pow(abs(b + point.x * k - point.y), 2);
}

double calc_loss(const std::vector<cv::Point2f>& points, double w1, double w2){
    // MSE
    double sse = 0;
    for (auto const& point: points) {
        double se = calc_error_y2(point, w2, w1);
        sse += se;
    }
    double mes = sse / (2 * points.size());
    return mes;

}

/// use the Eigen libary to implement the least square method
/// @param points a set of observed data points
/// @param k_ k of model parameters which best fit the data
/// @param b_ b of model parameters which best fit the data
/// @return whether a good model has been found
bool least_squares(const std::vector<cv::Point2f>& points, float& k_, float& b_){
    Eigen::MatrixXd A(points.size(), 2);
    Eigen::VectorXd b(points.size());
    
    int i = 0;
    for(auto const& point : points){
        A.row(i) = Eigen::Vector2d(point.x, 1);
        b(i) = point.y;
        i++;
    }
    // Eigen provides multiple solver
    // Eigen::MatrixXd solution = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    Eigen::VectorXd solution = A.colPivHouseholderQr().solve(b);

    if(solution.size()  == 2){
        k_ = solution(0);
        b_ = solution(1);
        return true;
    }
    return false;
}


/// use the Eigen libary to implement the least square method
/// @param points a set of observed data points
/// @param k_ k of model parameters which best fit the data
/// @param b_ b of model parameters which best fit the data
/// @return whether a good model has been found
bool least_squares_b(const std::vector<cv::Point2f>& points, float& k_, float& b_){
    Eigen::MatrixXd A(points.size(), 2);
    Eigen::VectorXd b(points.size());
    
    int i = 0;
    for(auto const& point : points){
        A.row(i) = Eigen::Vector2d(point.x, 1);
        b(i) = point.y;
        i++;
    }
    
    Eigen::MatrixXd ATA = A.transpose() * A;
    auto rank =  Eigen::FullPivLU<Eigen::MatrixXd>(ATA).rank();
    if(rank < ATA.rows()){
        std::cout << "rank of A^TA is " << rank << ", can't directly get inverse" << std::endl;
        return false;
    }
    
    Eigen::VectorXd solution = ATA.inverse() * A.transpose() * b;
    if(solution.size()  == 2){
        k_ = solution(0);
        b_ = solution(1);
        return true;
    }
    return false;
}


/// implement the least square method from [there](https://blog.csdn.net/pl20140910/article/details/51926886)
/// @param points a set of observed data points
/// @param k_ k of model parameters which best fit the data
/// @param b_ b of model parameters which best fit the data
/// @return whether a good model has been found
bool least_squares_a(const std::vector<cv::Point2f>& points, float& k_, float& b_){
    int n = (int)points.size();
    float A=0,B=0,C=0,D=0;
    for(auto const& point : points){
        A += point.x * point.x;
        B += point.x;
        C += point.x * point.y;
        D += point.y;
    }
    
    float temp = (n*A - B*B);
    if(temp){
        k_ = (n*C - B*D) / temp;
        b_ = (A*D - B*C) / temp;
        return true;
    }else{
        return false;
    }
}


/// standard RANSAC algorithm from [wikipedia](https://en.wikipedia.org/wiki/Random_sample_consensus)
/// @param data a set of observed data points
/// @param k_ k of model parameters which best fit the data
/// @param b_ b of model parameters which best fit the data
/// @param n the minimum number of data values required to fit the model
/// @param k the maximum number of iterations allowed in the algorithm
/// @param t a threshold value for determining when a data point fits a model
/// @param d the number of close data values required to assert that a model fits well to data
/// @return whether a good model has been found
bool ransac(const std::vector<cv::Point2f>& data, float& k_, float& b_, int n = 10, int k = 1000, float t = 10, int d = 10){
    int c = (int)data.size();
    std::default_random_engine e(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<> u(0, c-1);
    
    float best_error = FLT_MAX;
    float best_model_k = 0, best_model_b = 0;
    
    std::vector<int> best_indexs;
    int it = 0;
    while (it++ < k) {
        // n randomly selected values from data
        int maybe_it = 0;
        std::vector<bool> visited(c, false);
        std::vector<cv::Point2f> maybe_inliers;
        while(true && maybe_it++ < k ){
            int i = u(e);
            if(visited[i]) continue;
            
            maybe_inliers.push_back(data[i]);
            visited[i] = true;
            
            if(maybe_inliers.size() == n) break;
        }
        if(maybe_inliers.empty()) return false;
        
        // model parameters fitted to maybeinliers
        float maybe_model_k = 0, maybe_model_b = 0;
        least_squares(maybe_inliers, maybe_model_k, maybe_model_b);
        std::vector<cv::Point2f> also_inliers;
        
        for(int i = 0; i < c; i++){
            if(visited[i]) continue;// every point in data not in maybeinliers
            float error = calc_error(data[i], maybe_model_k, maybe_model_b);
            if(error > t) continue;//point fits maybemodel with an error smaller than t
            also_inliers.push_back(data[i]);//add point to alsoinliers
        }
//        std::cout << also_inliers.size() << std::endl;
        if(also_inliers.size() <= d) continue;//the number of elements in alsoinliers is > d
        
        // this implies that we may have found a good model
        // now test how good it is
        
        // parameters fitted to all points in maybeinliers and alsoinliers
        also_inliers.insert(also_inliers.end(), maybe_inliers.begin(), maybe_inliers.end());
        float better_model_k = 0, better_model_b = 0;
        least_squares(also_inliers, better_model_k, better_model_b);
        
        // measure of how well model fits these points
        float this_error = 0;
        for(int i = 0; i < also_inliers.size(); i++){
            this_error += calc_error(also_inliers[i], better_model_k, better_model_b);
        }
        this_error = this_error/also_inliers.size();
        
        
        bool better = this_error < best_error;
        if(better){
            best_model_k = better_model_k;
            best_model_b = better_model_b;
            best_error = this_error;
        }
    }
    
    k_ = best_model_k;
    b_ = best_model_b;
    return best_error != FLT_MAX;
}

/// gradient descent
/// @param data a set of observed data points
/// @param k_ k of model parameters which best fit the data
/// @param b_ b of model parameters which best fit the data
/// @param alpha learing rate
/// @param k the maximum number of iterations allowed in the algorithm
/// @param min_loss  the loss of close data values required to assert that a model fits well to data
bool gradient_descent(const std::vector<cv::Point2f>& data, float& k_, float& b_, double alpha = 0.1, int k = 1000, double min_loss = 0.001){
    int c = (int)data.size();
    double loss = FLT_MAX;
    double loss_pre = loss;
    
    int it = 0;
    double w1 = 0, w2 = 0;
    do {
        // the partial derivative of w1,w2
        double g1 = 0, g2 = 0;
        for(int i=0; i<c; i++){
            g1 = g1 + w1 + w2 * data[i].x - data[i].y;
            g2 = g2+ (w1  + w2 * data[i].x  - data[i].y) * data[i].x;
        }
        
        w1 = w1 - alpha * g1 / c;// update w1
        w2 = w2 - alpha * g2 / c;// update w2
        
        loss_pre = loss;
        loss = calc_loss(data, w1, w2);
        
    }while(abs(loss - loss_pre) >= min_loss && it++ < k);
    
    
    // std::cout<<"gradient_descent loss it:" << it << " loss:"<< loss << std::endl;
    k_ = w2;
    b_ = w1;
    return loss != FLT_MAX && !isinf(loss);
}

int main(int argc, const char * argv[]) {
    std::cout << std::fixed << std::setprecision(8);
    
    float k = 0.5;
    float b = 20;
    cv::Mat board = cv::Mat(800, 800, CV_8UC3);
    
    std::vector<cv::Point2f> points = generate_data(k, b, 300, 10, 400, true);
    
    std::cout<< "original   k:" << k << " b:" << b << std::endl;
    draw_data(board, points);
    draw_line(board, k, b, cv::Scalar(255,255,255));
    
    float k_, b_;
//    if(least_squares_a(points, k_, b_)){
//        std::cout<< "least squares a:" << k_ << " b:" << b_ << std::endl;
//        draw_line(board, k_, b_, cv::Scalar(0,255,0));
//    }
//
//    if(least_squares_b(points, k_, b_)){
//        std::cout<< "least squares b:" << k_ << " b:" << b_ << std::endl;
//        draw_line(board, k_, b_, cv::Scalar(0,255,0));
//    }

    if(least_squares(points, k_, b_)){
          std::cout<< "least squares:" << k_ << " b:" << b_ << std::endl;
          draw_line(board, k_, b_, cv::Scalar(0,255,0));
      }


    if(ransac(points, k_, b_, 10, 1000, 50, 100)){
        std::cout<< "ransac k:" << k_ << "b:" << b_ << std::endl;
        draw_line(board, k_, b_, cv::Scalar(0,0,255));
    }

    if(gradient_descent(points, k_, b_, 0.00004, 1e8, 1e-8)){
        std::cout<< "gradient_descent k:" << k_ << "b:" << b_ << std::endl;
        draw_line(board, k_, b_, cv::Scalar(0,255,255));
    }

    cv::imshow("board", board);
    cv::waitKey();
    return 0;
}
