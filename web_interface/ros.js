// Connecting to ROS
// -----------------
var beginning = Date.now();
var ros = new ROSLIB.Ros();

// Guess connection of the rosbridge websocket
function getRosBridgeHost() {
  if (window.location.protocol == 'file:') {
    return '192.168.1.3';
    //return 172.27.158.58
    //return '192.168.1.10';
  } else {
    return window.location.hostname;
  }
}

var rosBridgePort = 9090;
// Create a connection to the rosbridge WebSocket server.
ros.connect('ws://' + getRosBridgeHost() + ':' + rosBridgePort);

// First, we create a Topic object with details of the topic's name and message type.
var logTopic = new ROSLIB.Topic({
  ros : ros,
  name : '/web_interface/log',
  messageType : 'std_msgs/String'
});

// First, we create a Topic object with details of the topic's name and message type.
var elemPressed = new ROSLIB.Topic({
  ros : ros,
  name : '/hrc_learning/web_interface/pressed',
  messageType : 'std_msgs/String'
});

// Topic for error passing to the left arm
var errorPressedL = new ROSLIB.Topic({
  ros : ros,
  name : '/robot/digital_io/left_lower_button/state',
  messageType : 'baxter_core_msgs/DigitalIOState'
});

// Topic for error passing to the right arm
var errorPressedR = new ROSLIB.Topic({
  ros : ros,
  name : '/robot/digital_io/right_lower_button/state',
  messageType : 'baxter_core_msgs/DigitalIOState'
});

var rightArmInfo = new ROSLIB.Topic({
    ros : ros,
    name: '/action_provider/right/state',
    messageType : 'human_robot_collaboration_msgs/ArmState'
});

var leftArmInfo = new ROSLIB.Topic({
    ros : ros,
    name: '/action_provider/left/state',
    messageType : 'human_robot_collaboration_msgs/ArmState'
});

// Add a callback for any element on the page
function callback(e) {
    var e = window.e || e;

    if (e.target.tagName == 'BUTTON')
    {
        var obj = e.target.firstChild.nodeValue;
        var message = new ROSLIB.Message({
            data: obj
        });

        elemPressed.publish(message);
    }
    return;
}

if (document.addEventListener)
    document.addEventListener('click', callback, false);
else
    document.attachEvent('onclick', callback);
