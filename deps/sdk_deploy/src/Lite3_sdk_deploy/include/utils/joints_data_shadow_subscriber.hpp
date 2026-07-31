#pragma once

#include <memory>

#include "drdds/msg/joints_data.hpp"
#include "rclcpp/rclcpp.hpp"
#include "topic_trace.hpp"

class JointsDataShadowSubscriber {
public:
    explicit JointsDataShadowSubscriber(const rclcpp::Node::SharedPtr& node) : node_(node) {
        if (!node_) {
            throw std::invalid_argument("JointsDataShadowSubscriber requires a valid ROS2 node");
        }

        joints_data_sub_ = node_->create_subscription<drdds::msg::JointsData>(
            "/JOINTS_DATA",
            10,
            std::bind(&JointsDataShadowSubscriber::Callback, this, std::placeholders::_1));

        RCLCPP_INFO(node_->get_logger(), "Shadow subscriber attached to /JOINTS_DATA");
    }

private:
    void Callback(const drdds::msg::JointsData::SharedPtr msg) {
        topic_trace::LogEvent(
            node_->get_logger(),
            joints_data_shadow_trace_,
            "SUB /JOINTS_DATA_SHADOW",
            rclcpp::Time(msg->header.stamp).seconds(),
            msg->header.frame_id);

        if (!first_message_seen_) {
            first_message_seen_ = true;
            RCLCPP_INFO(
                node_->get_logger(),
                "Shadow subscriber received first /JOINTS_DATA: seq=%llu joint0_pos=%.6f joint0_vel=%.6f",
                static_cast<unsigned long long>(msg->header.frame_id),
                msg->data.joints_data[0].position,
                msg->data.joints_data[0].velocity);
        }
    }

    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<drdds::msg::JointsData>::SharedPtr joints_data_sub_;
    topic_trace::TraceState joints_data_shadow_trace_;
    bool first_message_seen_{false};
};
