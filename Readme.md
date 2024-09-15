## Ancora 1.0

### Purpose

- A multi-agent system for learning, navigating and generalizing large environments

Coding guidelines:

- all files with lowercase and underscores
- Naming should be done from the most abstract and common elements to the most specific and concrete. Treat naming as a
  meta language, placing where the action happens and what happens
    - e.g. `storage_add_item` places you the the abstract storage space and tells you what happens there
    - Using the same example, `add_item_storage` would tell you an action which can happen in multiple places
      potentially. If add_item takes place only in storage its fine, but ideally you want to specify the module/space of
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

There will be a number of "LocalDoc" files in the project with the purpose of providing a quick overview of the module.