#ifndef __DECOMP_CONSTRAINTS_H_
#define __DECOMP_CONSTRAINTS_H_

// #include <mpc_planner_modules/controller_module.h>
#include <ros/ros.h>
#include <tf/transform_listener.h> // For handling transforms
#include <costmap_2d/costmap_2d_ros.h>
#include <std_msgs/Empty.h>
#include <std_srvs/Empty.h>

#include <decomp_util/ellipsoid_decomp.h>
#include <decomp_util/decomp_geometry/geometric_utils.h>

#include <ros_tools/projection.h>

#include <mpc_planner_standalone/decomp.h> // Change this to your actual package name

#include <memory.h>

class DecompNode
{
public:
  DecompNode();

public:
  bool computeConstraintsCallback(mpc_planner_standalone::decomp::Request &req,
                                  mpc_planner_standalone::decomp::Response &res);
  // void update(State &state, const RealTimeData &data, ModuleData &module_data) override;
  // void setParameters(const RealTimeData &data, const ModuleData &module_data, int k) override;

  // bool isDataReady(const RealTimeData &data, std::string &missing_data) override;

  void resetCallback(const std_msgs::Empty &msg);
  void visualize();

  void occupancyGridCallback(costmap_2d::Costmap2D &msg);

private:
  ros::NodeHandle nh_;
  ros::ServiceServer service_;
  std::shared_ptr<costmap_2d::Costmap2DROS> costmap_ros_, global_costmap_ros_;
  std::unique_ptr<tf2_ros::Buffer> _tf_buffer;
  std::unique_ptr<tf2_ros::TransformListener> _tf_listener;
  std::vector<std::vector<Eigen::ArrayXd>> _a1, _a2, _b; // Constraints [disc x step]
  ros::ServiceClient _clear_client;
  ros::Subscriber _reset_sub;

  std::unique_ptr<EllipsoidDecomp2D> _decomp_util;
  vec_Vec2f _occ_pos;
  std::vector<LinearConstraint<2>> _constraints; // Static 2D halfspace constraints set in DecompUtil
  vec_E<Polyhedron<2>> _polyhedrons;
  std::vector<std::unique_ptr<vec_Vec2f>> occ_pos_vec_stages_;

  double _dummy_a1{1.}, _dummy_a2{0.}, _dummy_b;

  int _n_discs;

  MPCPlanner::DouglasRachford dr_projection_;

  int _max_constraints;

  ros::Subscriber _occ_sub;

  bool getOccupiedGridCells();

  void projectToSafety(Eigen::Vector2d &pos);
};

#endif // __DECOMP_CONSTRAINTS_H_
