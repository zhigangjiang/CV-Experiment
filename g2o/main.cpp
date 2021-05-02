#include <stdio.h>
#include <iostream>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/slam2d/types_slam2d.h>
#include <Eigen/Core>

typedef struct{
    int s, e;
    Eigen::Vector2d pose;
} Edge;
typedef g2o::BlockSolver< g2o::BlockSolverTraits<2, 2> >  SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

int main(int argc, const char * argv[]) {

    std::vector<Edge> edgeData = {
        {0, 1, {1, 1}},
        {1, 2, {1, -1}},
        {2, 3, {-1, -1}},
        {3, 0, {-0.5, 0.5}},
    };
    
    std::unique_ptr<SlamLinearSolver> linearSolver = g2o::make_unique<SlamLinearSolver>();
    linearSolver->setBlockOrdering(false);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<SlamBlockSolver>(std::move(linearSolver)));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    auto maxEdge = std::max_element(edgeData.begin(), edgeData.end(), [](const Edge& a, const Edge& b){
        return std::max(a.e, a.s) < std::max(b.e, b.s);
    });
    int maxIndex = std::max(maxEdge->s, maxEdge->e);

    for(int i = 0; i < maxIndex+1; i++){
        g2o::VertexPointXY *v = new g2o::VertexPointXY();
        v->setId(i);
        v->setEstimate(g2o::Vector2());
        if(i == 0){
            v->setFixed(true);
        }
        optimizer.addVertex(v);
    }

    for(const auto& pData: edgeData){
        g2o::EdgePointXY* edge = new g2o::EdgePointXY();
        edge->setVertex( 0, optimizer.vertex(pData.s));
        edge->setVertex( 1, optimizer.vertex(pData.e));
        edge->setInformation(  Eigen::Matrix< double, 2,2 >::Identity() );// 信息矩阵表示2维上侧重哪一维，xy是一样重要的，所以就是单位矩阵，但是在6维的位姿中，有可能更侧重优化旋转或者位移，就需要设置信息矩阵
        edge->setMeasurement(pData.pose );
        optimizer.addEdge(edge);
    }
    
    optimizer.initializeOptimization();
    optimizer.optimize(500);
    for(int i = 0; i < maxIndex+1; i++){
        g2o::VertexPointXY* vertex = dynamic_cast<g2o::VertexPointXY*>(optimizer.vertex( i ));
        g2o::Vector2 pose = vertex->estimate();
        std::cout << i << ":\n" << pose << std::endl ;
    }
    return 0;
}
