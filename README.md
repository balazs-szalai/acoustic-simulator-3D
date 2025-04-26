# Acoustic simulator 3D
This is a 3D acoustic simulator. It is written in Python using Taichi.
You can build simple geometries with the model builder, add sound sources and then calculate the time dependent wave equation. 
The geometries are then converted into a square mesh, this is where the wave equation is solved. You can define materials and build the models from them.
It is fun to use for visualization, but isn't as good for measurement simulation because for the implementation of absorbing walls the 1st approximation of the Mur's absorbing boundary condition was used, which is not perfect for waves comoing under an angle.
Feel free to improve this, I just abandoned it.
