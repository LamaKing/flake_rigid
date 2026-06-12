Depinning dynamics
============================

The previous examples dealt with static properties of the interface.
``flake.dynamics.run_md`` integrates the overdamped equation of motion (see :doc:`intro`) and follows the trajectory of a cluster in time.
See ``examples/4-Dynamics.ipynb`` for a worked example.

An example of interesting time evolution is the depinning of a cluster under external drivers: a torque and a force applied to its center of mass (CM).
If these drivers exceed the critical threshold, the cluster depins and begins to translate and rotate.

The animation below shows the cluster evolving during depinning, with each particle coloured according to its potential energy. The applied torque is counter-clockwise and the force is along :math:`x`.

.. figure:: _static/trajectory.gif
            :height: 400px

            Depinning cluster under torque and force. Particle colour indicates substrate potential energy (blue: low, yellow: high). Arrows show the force on each particle (length proportional to magnitude).

From the trajectory we can plot the total energy of the cluster as a function of time.

.. figure:: _static/energy.png
           :height: 400px

           Energy per particle as a function of time.


Because the drivers in this example are only slightly above the critical value, the motion is *intermittent* — reminiscent of `stick-slip <https://en.wikipedia.org/wiki/Stick-slip_phenomenon>`_ (strictly speaking, true stick-slip cannot occur in a rigid system under a constant force).
In the limit of large drives, the cluster moves in an almost-smooth fashion as the substrate force becomes negligible.

.. figure:: _static/x_trajectory.png
           :height: 400px

           Position of the cluster CM along :math:`x` as a function of time.


Note that the amplitude of the "slips" decreases with time, converging toward smooth sliding.
This is due to the coupling of rotation and translation: as the cluster rotates, the effective depth of the energy landscape decreases, making the applied force proportionally larger relative to the barrier — pushing the system closer to the smooth-sliding limit.
