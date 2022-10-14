#pragma once

#include <stdlib.h>
#include <set>
#include <random>
#include "../../RaisimGymEnv.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <math.h>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {
    public:
        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
            world_ = std::make_unique<raisim::World>();
            world_->addGround(0, "steel");
            world_->setERP(1.0);

            arm_ = world_->addArticulatedSystem(resourceDir_ + "/arm_round.urdf");
            arm_->setName("arm");
            arm_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->setMaterialPairProp("default", "steel", 5.0, 0.0, 0.0001);

            gcDim_ = arm_->getGeneralizedCoordinateDim();
            gvDim_ = arm_->getDOF();
            nJoints_ = gvDim_ - 6;

            gc_.setZero(gcDim_); gc_init_.setZero(gcDim_); gc_ref_.setZero(gcDim_);
            gv_.setZero(gvDim_); gv_init_.setZero(gvDim_); gv_ref_.setZero(gvDim_);

            gc_init_ <<
                0, 0, 0.05, // base pos 0
                1, 0, 0, 0, // base rot 3
                1, 0, 0, 0, // shoulder rot 7
                0, // elbow rot 11
                1, 0, 0, 0; // wrist rot 12
            gc_ref_ <<
                0, 0, 0.05, // base pos 0
                1, 0, 0, 0, // base rot 3
                1, 0, 0, 0, // shoulder rot 7
                0, // elbow rot 11
                1, 0, 0, 0; // wrist rot 12

            ball_ = world_->addArticulatedSystem(resourceDir_ + "/ball3D.urdf");
            ball_->setName("ball");
            ball_->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);
            world_->setMaterialPairProp("default", "ball", 1.0, 0.01, 0.0001);
            // world_->setMaterialPairProp("default", "ball", 1.0, 0.8, 0.0001);
            world_->setMaterialPairProp("ball", "steel", 5.0, 0.85, 0.0001);

            ball_gcDim_ = ball_->getGeneralizedCoordinateDim();
            ball_gvDim_ = ball_->getDOF();

            ball_gc_.setZero(ball_gcDim_); ball_gc_init_.setZero(ball_gcDim_); ball_gc_init_[3] = 1;
            ball_gv_.setZero(ball_gvDim_); ball_gv_init_.setZero(ball_gvDim_);

            pTarget_.setZero(gcDim_);
            vTarget_.setZero(gvDim_);

            data_gc_ = Eigen::MatrixXd::Zero(dataLen_, data_gcDim_);

            read_data();

            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(5.0);

            arm_->setPdGains(jointPgain, jointDgain);
            arm_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            obDim_ = 41;
            // joint positions 3 + 3 + 3
            // joint velocities 3 + 3 + 3
            // joint orientations 4 + 1 + 4
            // joint angular velocities 3 + 1 + 3
            // ball position 3
            // ball velocity 3
            // phase variable 1
            obDouble_.setZero(obDim_);

            actionDim_ = gcDim_;

            stateDim_ = gcDim_ + 7;
            stateDouble_.setZero(stateDim_);
            
            rewards_.initializeFromConfigurationFile(cfg["reward"]);

            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(arm_);
            }
    }

    void init() final {}

    void read_data(){
        std::ifstream gcfile(resourceDir_ + "/walk_gc.txt");
        float data;
        int i = 0, j = 0;
        while (gcfile >> data){
            data_gc_.coeffRef(j, i) = data;
            i++;
            if (i >= gcDim_){
                i = 0;
                j++;
            }
        }
    }

    void reset() final {
        sim_step_ = 0;
        total_reward_ = 0;
        index_ = rand() % dataLen_;
        // gc_ref_.segment(7, 4) = data_gc_.row(index_).segment(14, 4);
        gc_ref_[11] = 1.57; //1.57;

        pTarget_ << gc_ref_;

        // std::cout << "PTARGET=======================" << std::endl;
        // std::cout << pTarget_ << std::endl;

        arm_->setState(gc_ref_, gv_ref_);

        Vec<3> right_hand_pos;
        size_t right_hand_idx = arm_->getFrameIdxByName("wrist");
        arm_->getFramePosition(right_hand_idx, right_hand_pos);
        ball_gc_init_[0] = right_hand_pos[0];
        ball_gc_init_[1] = right_hand_pos[1];
        ball_gc_init_[2] = right_hand_pos[2] - 0.15; // ball 0.11, hand 0.015
        ball_gv_init_[2] = 0.05;
        ball_->setState(ball_gc_init_, ball_gv_init_);

        from_ground_ = false;
        from_hand_ = false;
        is_ground_ = false;
        is_hand_ = false;
        ground_hand_ = false;
        terminal_flag_ = false;

        updateObservation();
    }

    void updateObservation() {
        arm_->getState(gc_, gv_);
        ball_->getState(ball_gc_, ball_gv_);

        // std::cout << "GC=============================" << std::endl;
        // std::cout << gc_ << std::endl;

        Mat<3,3> rootRotInv;
        Vec<3> rootPos;
        getRootTransform(rootRotInv, rootPos);

        Vec<3> jointPos_W, jointPos_B, jointVel_W, jointVel_B;
        int obIdx = 0;
        int gcIdx = 7;
        int gvIdx = 6;
        
        //  0 shoulder pos
        //  3 shoulder vel
        //  6 shoulder orn
        // 10 shoulder ang
        // 13 elbow pos
        // 16 elbow vel
        // 19 elbow orn
        // 20 elbow ang
        // 21 wrist pos
        // 24 wrist vel
        // 27 wrist orn
        // 31 wrist ang
        // 34 ball pos
        // 37 ball vel
        // 40 phase

        for(size_t bodyIdx = 1; bodyIdx < 4; bodyIdx++){
            arm_->getBodyPosition(bodyIdx, jointPos_W);
            arm_->getVelocity(bodyIdx, jointVel_W);
            matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
            matvecmul(rootRotInv, jointVel_W, jointVel_B);

            obDouble_.segment(obIdx, 3) = jointPos_B.e();
            obIdx += 3;
            obDouble_.segment(obIdx, 3) = jointVel_B.e();
            obIdx += 3;

            if (bodyIdx == 2) { // revolute jointIdx
                obDouble_.segment(obIdx, 1) = gc_.segment(gcIdx, 1);
                obIdx += 1; gcIdx += 1;
            }
            else {
                obDouble_.segment(obIdx, 4) = gc_.segment(gcIdx, 4);
                obIdx += 4; gcIdx += 4;
            }

            if (bodyIdx == 2) { // revolute jointIdx
                obDouble_.segment(obIdx, 1) = gv_.segment(gvIdx, 1);
                obIdx += 1; gvIdx += 1;
            }
            else {
                obDouble_.segment(obIdx, 3) = gv_.segment(gvIdx, 3);
                obIdx += 3; gvIdx += 3;
            }
        // std::cout << "OBIDX" << obIdx << std::endl;
        }
        
        
        // relative ball pos vel
        matvecmul(rootRotInv, ball_gc_.head(3) - rootPos.e(), jointPos_B);
        matvecmul(rootRotInv, ball_gv_.head(3), jointVel_B);
        obDouble_.segment(obIdx, 3) = jointPos_B.e();
        obIdx += 3;
        obDouble_.segment(obIdx, 3) = jointVel_B.e();

        obDouble_.tail(1) << phase_;
    }

    void getRootTransform(Mat<3,3>& rot, Vec<3>& pos) {
        double yaw = atan2(2 * (gc_[3] * gc_[4] + gc_[5] * gc_[6]), 1 - 2 * (gc_[4] * gc_[4] + gc_[5] * gc_[5]));
        Vec<4> quat;
        quat[0] = cos(yaw / 2); quat[1] = 0; quat[2] = 0; quat[3] = - sin(yaw / 2);
        raisim::quatToRotMat(quat, rot);
        pos[0] = gc_[0]; pos[1] = gc_[1]; pos[2] = gc_[2];
    }

    float step(const Eigen::Ref<EigenVec>& action) final {
        int actionIdx = 3;
        // std::cout << "PTARGETBEFORE=======================" << std::endl;
        // std::cout << pTarget_ << std::endl;
        for (int j = 0; j < 4; j++)
        {
            if (j == 2) {
                pTarget_.segment(actionIdx, 1) << pTarget_.segment(actionIdx, 1) + action.cast<double>().segment(actionIdx, 1);
                // pTarget_.segment(actionIdx, 1) << action.cast<double>().segment(actionIdx, 1);
                actionIdx += 1;
            }
            else{
                pTarget_.segment(actionIdx, 4) << pTarget_.segment(actionIdx, 4) + action.cast<double>().segment(actionIdx, 4);
                // pTarget_.segment(actionIdx, 4) << action.cast<double>().segment(actionIdx, 4);
                pTarget_.segment(actionIdx, 4) << pTarget_.segment(actionIdx, 4).normalized();
                actionIdx += 4;
            }
        }

        // std::cout << "ACTION=============================" << std::endl;
        // std::cout << action << std::endl;
        // std::cout << "PTARGETAFTER=======================" << std::endl;
        // std::cout << pTarget_ << std::endl;

        arm_->setPdTarget(pTarget_, vTarget_);

        for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++)
        {
            if (server_) server_->lockVisualizationServerMutex();
            world_->integrate();
            if (server_) server_->unlockVisualizationServerMutex();

            for (auto& contact: ball_->getContacts()){
                if (contact.getPosition()[2] < 0.01)
                {
                    if (from_ground_) {
                        // std::cout << "GG" << std::endl;
                        terminal_flag_ = true;
                        break;
                    }
                    if (is_hand_) {
                        // std::cout << "DOUBLE" << std::endl;
                        terminal_flag_ = true;
                        break;
                    }
                    is_ground_ = true;
                    from_ground_ = true;
                    from_hand_ = false;
                }
                else {
                    auto& pair_contact = world_->getObject(contact.getPairObjectIndex())->getContacts()[contact.getPairContactIndexInPairObject()];
                    if (arm_->getBodyIdx("wrist") == pair_contact.getlocalBodyIndex()){
                        if (is_ground_) {
                            // std::cout << "DOUBLE" << std::endl;
                            terminal_flag_ = true;
                            break;
                        }
                        if (from_ground_) {
                            ground_hand_ = true;
                        }
                        is_hand_ = true;
                        from_hand_ = true;
                        from_ground_ = false;
                    }
                    else {
                        // std::cout << "OTHER BODY CONTACT" << std::endl;
                        terminal_flag_ = true;
                        break;
                    }
                }
            }
        }

        is_hand_ = false;
        is_ground_ = false;

        updateObservation();
        computeReward();

        double current_reward = rewards_.sum();
        total_reward_ += current_reward;
        return current_reward;
    }

    void computeReward() {
        double ball_dist = (obDouble_[21] - obDouble_[34]) * (obDouble_[21] - obDouble_[34]) + (obDouble_[22] - obDouble_[35]) * (obDouble_[22] - obDouble_[35]);
        double dribble_reward = 0;
        double dist_reward = 0;
        if (ground_hand_) {
            dribble_reward += 1;
            ground_hand_ = false;
        }
        dist_reward += exp(-ball_dist);
        rewards_.record("dribble", dribble_reward);
        rewards_.record("ball distance", dist_reward);
    }

    void observe(Eigen::Ref<EigenVec> ob) final {
        ob = obDouble_.cast<float>();
    }
   
    bool time_limit_reached() {
        return sim_step_ > max_sim_step_;
    }
    
    float get_total_reward() {
        return float(total_reward_);
    }

    bool isTerminalState(float& terminalReward) final {
        if (time_limit_reached()) {
            // std::cout << "max sim step" << std::endl;
            return true;
        }
        // ball too far
        // if ((ball_gc_[0] - gc_[0]) * (ball_gc_[0] - gc_[0]) + (ball_gc_[1] - gc_[1]) * (ball_gc_[1] - gc_[1]) > 1) {
        //     return true;
        // }
        if ((obDouble_[21] - obDouble_[34]) * (obDouble_[21] - obDouble_[34]) + (obDouble_[22] - obDouble_[35]) * (obDouble_[22] - obDouble_[35]) > 1){
            // std::cout << "ball too far" << std::endl;
            return true;
        }
        if (terminal_flag_)
        {
            // std::cout << "else" << std::endl;
            return true;
        }
        return false;
    }

    void getState(Eigen::Ref<EigenVec> ob) final {
        stateDouble_ << gc_.tail(gcDim_), ball_gc_.tail(7);
        ob = stateDouble_.cast<float>();
    }

    private:
        bool visualizable_ = false;
        raisim::ArticulatedSystem* arm_;
        raisim::ArticulatedSystem* ball_;

        int gcDim_, gvDim_, nJoints_;
        int ball_gcDim_, ball_gvDim_;
        int data_gcDim_ = 43;
        int data_gvDim_ = 34;

        Eigen::VectorXd ball_gc_, ball_gv_, ball_gc_init_, ball_gv_init_;
        Eigen::VectorXd gc_, gv_, gc_init_, gv_init_, gc_ref_, gv_ref_;

        int dataLen_ = 39;
        int index_ = 0;
        Eigen::MatrixXd data_gc_;

        Eigen::VectorXd pTarget_, vTarget_;

        Eigen::VectorXd obDouble_, stateDouble_;

        float phase_ = 0;
        float phase_speed = 0;
        float max_phase = 1;
        int sim_step_ = 0;
        int max_sim_step_ = 1000;
        double total_reward_ = 0;

        bool terminal_flag_ = false;
        
        bool from_hand_ = false;
        bool from_ground_ = false;
        bool is_ground_ = false;
        bool is_hand_ = false;
        bool ground_hand_ = false;
};
}