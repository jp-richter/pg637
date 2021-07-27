#pragma once

#include <string>
#include <algorithm>

#include "Tools/Joints.h"

namespace Config
{{
  constexpr int NUM_RELEVANT_JOINTS = 12;

  const int algorithmId = 0;
  const int randomSeed = 9393;

  /*
  * Add joints here that you want the actor to train on. The actionSize, 
  * logging of actions and network shape will be determined automatically.
  * 
  * In the current setting rAnkleRoll will be set to -rHipRoll if rHipRoll
  * is contained in this array and rAnkleRoll is not (see LimbCombinator).
  */

  const std::array<Joints::Joint, 3> outputJointOrder
  {{
    Joints::rHipPitch,
    Joints::rAnklePitch,
   }};

  const std::array<Joints::Joint, NUM_RELEVANT_JOINTS> stateJoints
  {{
    Joints::firstArmJoint,
    Joints::firstRightArmJoint,
    Joints::lHipRoll,
    Joints::lHipPitch,
    Joints::lKneePitch,
    Joints::lAnklePitch,
    Joints::lAnkleRoll,
    Joints::rHipRoll,
    Joints::rHipPitch,
    Joints::rKneePitch,
    Joints::rAnklePitch,
    Joints::rAnkleRoll,
   }};

  const int stateSize = stateJoints.size() * 2 + 5;
  const int actionSize = outputJointOrder.size();
  const int actionInterval = 1;

  /*
  * With timing 90 80 the agent can do 7 actions per trajectory
  * 
  * TODO instead of actionInterval use vector of frames where robot should act
  */

  const int rewardFunctionId = 0;
  const float dampeningFactor = 0.5f;

  const int preKickDuration = 96;
  const int kickDuration = 84;
  const int postKickDuration = 1560;
  const int resetDuration = 100;

  const int iterations = 250;
  const float gamma = 0.99f;
  const int batchSize = {batchSize};
  const int hiddenSize = 32;
  const int bufferSize = 100000;
  const bool useCheckpoint = false;

  const float lrActor = {lrActor};
  const float lrCritic = {lrCritic};

  const float startStd = {startStd};
  const float endStd = -1.6f;
  const float epsilon = {epsilon};
  const int trajectoriesPerIteration = 128;
  const int sampleUsage = {sampleUsage};
  const int bootstrapK = 4;

  const std::string shortDescription =
    "A halfed lr";
  const std::string longDescription =
    "";
  const std::string experimentName =
    "APPO-HipAnkle-A";

  struct Range {{
    float min;
    float max;
  }};

  const std::map<Joints::Joint, std::string> jointToLogName
  {{
       {{Joints::rHipPitch, "rHipPitch"}},
       {{Joints::rHipRoll, "rHipRoll"}},
       {{Joints::rKneePitch, "rKneePitch"}},
       {{Joints::rAnklePitch, "rAnklePitch"}},
       {{Joints::rAnkleRoll, "rAnkleRoll"}}
   }};

  /*
  * Below are the angle ranges of the nao robot joints (commented). 
  * The ranges in the actual map are chosen somewhat arbitrarily to be 
  * not restrictive in any kind but missing unreasonable values. 
  * 
  * For reference see
  * https://github.com/NaoDevils/NDevils2015/issues/349
  * http://doc.aldebaran.com/2-8/family/nao_technical/joints_naov6.html
  */  
  const std::map<Joints::Joint, Range> jointToRange
  {{
       {{ Joints::rHipPitch, {{-1.3f, 0.0f}} }},
       {{ Joints::rHipRoll, {{-0.3f, 0.0f}} }},
       {{ Joints::rKneePitch, {{-0.1f, 1.5f}} }},
       {{ Joints::rAnklePitch, {{-1.0f, 0.0f}} }},
       {{ Joints::rAnkleRoll, {{-0.0f, 0.3f}} }}
   }}; 

   }}
