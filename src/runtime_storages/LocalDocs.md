The storage module uses a system similar to rust. Each storage struct instantiated can be thought of as an object on
with certain operations that work based on it
The reason we didn't employ objects is because the ties between state and functions lead to an awkward situation where:

- If too many function accumulate you have to split them into multiple objects which introduce unnecessary boundaries
  and complexity
- If you want to split the functions outside and inject them as dependencies, the IDE won't be able to help you with
  autocompletion and you will have to remember them (which is bad for anyone using it). The solution is create a params
  object which introduces overhead when you call the function which is bad again
- If you keep all the functions in the same place you get a god object which is hard to manage and develop on

In other words I needed a way to organize a large number of methods which should in practice be grouped and used under
the same entity.
I deemed the structs to be the most appropriate since it separates the function from the state while still allowing the
functions to be used under the same imported module.
The only added overhead is passing the struct as the first argument to the function which is a small price to pay
compared to speed of development or even worse, the complexity of the codebase.