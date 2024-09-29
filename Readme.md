## Neuronav

## !!! UNDER REFACTORING !!!

### Purpose

- A multi-network system for learning, navigating large unstructured closed environments, with the capability to adapt
  and resilience to noise and external disturbances.

### Setup

Make sure to install requirements with `pip install -r requirements.txt` and follow the guidelines from [manim-community
website](https://docs.manim.community/en/stable/index.html) if you also want to run the manim visualizations.

### Coding guidelines:

Naming conventions:

- Files and modules should be named with snake_case
- Classes or types should be named with CamelCase
- Functions should be named with snake_case
- Variables should be named with snake_case
- Constants should be named with UPPER_SNAKE_CASE
- Modules should be named with snake_case

General guidelines to naming functions

- When naming a function, you should try with each word to convey as much information as possible, with the full name
  having a clear and easy to understand meaning.
    - Think of the name as a meta-language which is drawing boundaries and placing you in a narrow and clear context.
    - Try to maximize the amount of information given especially in the first words. If you have a query heavy
      application, using `get` in the beginning does not give much information since most functions will be queries.
      Starting instead with the context of the query such as `connections_get`, will place you in a narrow context,
      given that connections are a specific part of the application. On the other hand, if the app works with
      connections everywhere, while get operations are rare, `get_connections` would be more appropriate.
    - Keep in mind the module context. If you are calling the `storage` module, there is no point to name a function
      `storage_add_item`. `add_item` is enough since the context is clear.
    - Don't follow strict rules, just do what makes the most sense in that specific situation while still trying to keep
      clear boundaries and uniformity
    - This approach is inspired by the idea
      of [Shannon information theory](https://en.wikipedia.org/wiki/Information_theory). The more information you can
      convey in the
      first words, the better.


- e.g. `storage_add_item` places you the abstract storage space and tells you what happens there
- Using the same example, `add_item_storage` would tell you an action which can happen in multiple places
  potentially. If add_item takes place only in storage it's fine, but ideally you want to specify the module/space of
  the modified state and then the modifications. Shorter names like `add_item` are fine if the context is clear and
  it is limited to a single module
- For example, if `detach_action` is possible in a single context, the agent, it is much more indicative to name a
  functon `detach_action` than `agent_detach_action`, despite `detach` being a verb
- The role of the name is to say what it does. This implies you know where or how the function operates. Feel free
  to break the rules above if it makes sense in a specific context
- The general philosophy should to place anything from function names, to classes, to variables, etc into a context
  and make anyone understand what it does in that context. All the rules above are just guidelines to help you do
  that. Starting a function or name with `string` conveys no information at all. A good way to think about this is
  shanon information theory. Try to convey as many bits of information as possible
- Global contexts/singletons are fine as long as they are kept minimal and in tight check. Don't make a local variable
  and throw 100 references in the entire project for the sake of avoiding a singleton.
- interfaces have the prefix `interface`
- classes have the prefix `class`
- abstract classes have the prefix `abstract`

Other tips:

- Keep stuff that is related as together as possible. General stuff should ideally be just above the level it is used
- Generally employing a struct based architecture seems to work the best, given the large amount of functions needed
  under the same context, without introducing any additional boundaries or objects. Class-like behavior can be achieved
  if the struct is passed as a reference to the functions that need it and the module exports all the functions needed
  under the same umbrella, but without the need to create a class

There will be a number of "LocalDocs" files in the project with the purpose of providing a quick overview of the module.