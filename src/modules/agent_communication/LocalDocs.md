Agent refers to any medium in which navigation can be implemented. It can be simulation or real life, it doesn't really
matter.
The communication interface is simple enough and should allow many implementations underneath such as websockets, MQTT,
ROS, etc.
The teleport types should used only for simulations (unless you made the biggest breakthrough of the century in physics)

The detach actions are available throughout the entire application, and the response is handled via a global data buffer