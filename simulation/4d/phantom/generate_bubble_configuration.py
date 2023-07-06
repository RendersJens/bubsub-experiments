from foam_ct_phantom import FoamPhantom as foam

foam.generate('bubble_configuration.h5', 12345, nspheres_per_unit=1000)