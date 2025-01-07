#include "mpc_planner_standalone/decomp_node.h"

// #include <mpc_planner_parameters.h>

#include <mpc_planner_util/parameters.h>
// #include <mpc_planner_util/data_visualization.h>

#include <ros_tools/profiling.h>
#include <ros_tools/visuals.h>
// #include <ros_tools/spline.h>

// #include <algorithm>

DecompNode::DecompNode()
{
  // Initialize the ROS node handle
  nh_ = ros::NodeHandle();

  VISUALS.init(&nh_);

  _decomp_util = std::make_unique<EllipsoidDecomp2D>();
  Configuration::getInstance().initialize(SYSTEM_CONFIG_PATH(__FILE__, "settings"));

  // Only look around for obstacles using a box with sides of width 2*range
  double range = 5.;
  _decomp_util->set_local_bbox(Vec2f(range, range));

  _occ_pos.reserve(1000); // Reserve some space for the occupied positions

  _n_discs = 1;

  _max_constraints = 16;
  _a1.resize(_n_discs);
  _a2.resize(_n_discs);
  _b.resize(_n_discs);
  for (int d = 0; d < _n_discs; d++)
  {
    _a1[d].resize(CONFIG["N"].as<int>());
    _a2[d].resize(CONFIG["N"].as<int>());
    _b[d].resize(CONFIG["N"].as<int>());
    for (int k = 0; k < CONFIG["N"].as<int>(); k++)
    {
      _a1[d][k] = Eigen::ArrayXd(_max_constraints);
      _a2[d][k] = Eigen::ArrayXd(_max_constraints);
      _b[d][k] = Eigen::ArrayXd(_max_constraints);
    }
  }

  _tf_buffer.reset(new tf2_ros::Buffer(ros::Duration(10)));
  _tf_listener.reset(new tf2_ros::TransformListener(*_tf_buffer));

  costmap_ros_.reset(new costmap_2d::Costmap2DROS("local_costmap", *_tf_buffer));
  costmap_ros_->start();

  _reset_sub = nh_.subscribe("/lmpcc/reset_environment", 1, &DecompNode::resetCallback, this);
  _clear_client = nh_.serviceClient<std_srvs::Empty>("/move_base/clear_costmaps");

  // global_costmap_ros_.reset(new costmap_2d::Costmap2DROS("global_costmap", *_tf_buffer));
  // global_costmap_ros_->start();
  // Advertise the service
  service_ = nh_.advertiseService("compute_constraints", &DecompNode::computeConstraintsCallback, this);

  ROS_INFO("Compute Constraints Service is ready.");
}

void DecompNode::resetCallback(const std_msgs::Empty &msg)
{
  costmap_ros_->pause();
  costmap_ros_->resetLayers();
  ros::Duration(0.5).sleep();
  costmap_ros_->resume();
}

// Callback function for the service
bool DecompNode::computeConstraintsCallback(mpc_planner_standalone::decomp::Request &req,
                                            mpc_planner_standalone::decomp::Response &res)
{

  // auto &benchmarker = BENCHMARKERS.getBenchmarker("decomp");
  // benchmarker.start();

  if (req.x.size() < 2 || req.y.size() < 2)
    return false;

  if (req.first)
  {
    getOccupiedGridCells();
    _decomp_util->set_obs(_occ_pos); // Set them
  }
  _dummy_b = req.x.back() + 100.;

  vec_Vec2f path;
  int N = req.x.size();
  for (int k = 0; k < N; k++)
  {
    Eigen::Vector2d pos(req.x[k], req.y[k]);
    projectToSafety(pos);
    path.emplace_back(pos(0), pos(1));
  }

  bool not_good = false;
  for (int k = 1; k < N; k++)
  {
    auto a = path[k - 1];
    auto b = path[k];
    double d = std::sqrt(std::pow(a(0) - b(0), 2.) + std::pow(a(1) - b(1), 2.));
    if (d > 1.5)
      not_good = true;
  }

  if (not_good) // If we cross an obstacle, compute from the first state
  {
    path.clear();
    for (int k = 0; k < N; k++)
    {
      Eigen::Vector2d pos = Eigen::Vector2d(req.x[0], req.y[0]) + (double)k * Eigen::Vector2d(req.x[1] - req.x[0], req.y[1] - req.y[0]) * 0.1;
      projectToSafety(pos);
      path.emplace_back(pos(0), pos(1));
    }
  }

  // ROS_INFO("Path ready");

  _decomp_util->dilate(path, 0, false);

  _decomp_util->set_constraints(_constraints, 0.); // Map is already inflated
                                                   // auto new_polys = _decomp_util->get_polyhedrons();
                                                   // _polyhedrons.push_back(new_polys.back());
  _polyhedrons = _decomp_util->get_polyhedrons();

  res.constraint_a1.resize(_max_constraints * (N - 1));
  res.constraint_a2.resize(_max_constraints * (N - 1));
  res.constraint_b.resize(_max_constraints * (N - 1));

  for (int k = 0; k < N - 1; k++)
  {
    const auto &constraints = _constraints[k];
    int start = k * _max_constraints;

    int i = 0;
    for (; i < std::min((int)constraints.A_.rows(), _max_constraints); i++)
    {
      if (constraints.A_.row(i).norm() < 1e-3 || constraints.A_(i, 0) != constraints.A_(i, 0)) // If zero or nan
      {
        break;
      }

      res.constraint_a1[start + i] = constraints.A_.row(i)[0];
      res.constraint_a2[start + i] = constraints.A_.row(i)[1];
      res.constraint_b[start + i] = constraints.b_(i);
    }

    for (; i < _max_constraints; i++)
    {
      res.constraint_a1[start + i] = _dummy_a1;
      res.constraint_a2[start + i] = _dummy_a2;
      res.constraint_b[start + i] = _dummy_b;
    }

    if ((int)constraints.A_.rows() > _max_constraints)
    {
      ROS_WARN_STREAM("Maximum number of decomp util constraints exceeds specification: " << constraints.A_.rows()
                                                                                          << " > " << _max_constraints);
    }
  }

  // ROS_INFO_STREAM(res.constraint_a1[i] << " x + " << res.constraint_a2[i] << " y <= " << res.constraint_b[i]);

  if (req.last)
  {
    visualize();
    _polyhedrons.clear();
  }
  // benchmarker.stop();
  // BENCHMARKERS.print();

  // LOG_MARK("DecompConstraints::update done");
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "compute_constraints_server");

  // Create an instance of the ComputeConstraintsServer class
  DecompNode decomp_node;

  // Enter a loop to keep the node running
  ros::spin();

  return 0;
}

bool DecompNode::getOccupiedGridCells()
{
  const auto &costmap = *(costmap_ros_->getCostmap());

  // Store all occupied cells in the grid map
  _occ_pos.clear();
  double x, y;
  for (unsigned int i = 0; i < costmap.getSizeInCellsX(); i++)
  {
    for (unsigned int j = 0; j < costmap.getSizeInCellsY(); j++)
    {
      if (costmap.getCost(i, j) == costmap_2d::FREE_SPACE)
        continue;

      costmap.mapToWorld(i, j, x, y);
      // LOG_INFO("Obstacle at x = " << x << ", y = " << y);

      _occ_pos.emplace_back(x, y);
    }
  }
  // LOG_VALUE("Occupied cells", _occ_pos.size());

  return true;
}

void DecompNode::projectToSafety(Eigen::Vector2d &pos)
{
  if (_occ_pos.empty()) // There is no anchor
    return;

  // #pragma omp parallel for num_threads(4)
  for (auto &obs : _occ_pos)
  {
    double radius = 0.325 + 0.25;

    dr_projection_.douglasRachfordProjection(pos, obs, _occ_pos[0], radius, pos);
  }
}

void DecompNode::visualize()
{
  // ROS_INFO("Visualizing decomputil");

  bool visualize_points = false;

  auto &publisher = VISUALS.getPublisher("free_space");
  auto &polypoint = publisher.getNewPointMarker("CUBE");
  polypoint.setScale(0.1, 0.1, 0.1);
  polypoint.setColor(1, 0, 0, 1);

  auto &polyline = publisher.getNewLine();
  polyline.setScale(0.1, 0.1);
  for (int k = 0; k < _polyhedrons.size(); k += 3)
  {
    const auto &poly = _polyhedrons[k];
    polyline.setColorInt(k, (int)_polyhedrons.size());

    const auto vertices = cal_vertices(poly);
    if (vertices.size() < 2)
      return;

    for (size_t i = 0; i < vertices.size(); i++)
    {
      if (visualize_points)
        polypoint.addPointMarker(Eigen::Vector3d(vertices[i](0), vertices[i](1), 0));

      if (i > 0)
      {
        polyline.addLine(
            Eigen::Vector3d(vertices[i - 1](0), vertices[i - 1](1), 0),
            Eigen::Vector3d(vertices[i](0), vertices[i](1), 0));
      }
    }
    polyline.addLine(Eigen::Vector3d(vertices.back()(0), vertices.back()(1), 0),
                     Eigen::Vector3d(vertices[0](0), vertices[0](1), 0));
  }

  publisher.publish();
}
