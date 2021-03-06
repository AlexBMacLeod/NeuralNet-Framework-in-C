# c-nn

I started this off for CUDA C because I didn't see any examples of neural
networks on Github implemented in CUDA C. I've since found them, but be
that as it may the programming is actually quite intricate and quite fun, 
especially all the moving parts, and different pieces and how to put everything
together.

As a side note the Convolutional Layers are not currently working.

As far as functionality goes, for something more professional CUDA C++
makes more sense. Initially, machine learning owing so much to parallel
programming, I thought that this would be an interesting project. 
After starting and stopping a couple of times I realized that while the 
parallel programming parts could be challenging one of the biggest 
challenges was simply how to structure the data, and present it in such
a way that a user can easily use the interface. So I decided to do two
versions, one in C then one in CUDA C. The C version being a bit of a
guinea pig, and a place to work on network architecture, then
such work having been finished, the CUDA C version would be purely about
parallel programming, and the more mathematical side of machine learning.
This is the C version.

I was reading another guys github on doing this in CUDA C++ and he created
three primary classes: a matrix class, a layer class, and a neural 
network class; or something along those lines. I thought that made
smart sense so I did something similar. Of course no classes in C so
for the layers and matrices I used typedefed structs, then for the last,
the neural network class, I used a doubly linked list. 
I know there are a number of different ways to go about it 
what I like about the doubly linked list is that forward and 
backward propagation make alot more sense. The layers
themselves are fairly connected, while forward prop is fairly 
straightforward needing only the previous layers output, backprop uses 
far more needs the next layers weight matrix and delta. So with the initialziation of the rows in a linkedlist format
I can easily pass each row the pointers it needs to backprop, and thus 
somewhat simplify things that way.


net_add_layer("relu", 256, 512);
