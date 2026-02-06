# Perspective: Understanding the C++ Memory Model

> This article is adapted from the original internal slides. Core technical content is preserved.
> The structure, wording, and visuals are optimized for GitHub-style learning and sharing.

---

## 1. Introduction

The C++ memory model provides a foundational understanding of how a program’s execution interacts with and utilizes system memory.

**Why this matters**:
- It explains how program data and operations map onto virtual and physical memory.
- It is essential for writing efficient, safe, and maintainable C++ code.
- It directly impacts correctness in dynamic memory management, concurrency, and system-level programming.

---

## 2. Memory Model of a Program’s Lifeline

### 2.1 Overview

#### 2.1.1 Memory Accessibility

- **Byte is the smallest addressable unit**.
- Every byte has a unique address.
- This enables precise read/write operations at exact memory locations.

---

#### 2.1.2 Process Memory Model

Each process has its own **virtual address space**, which is mapped by the OS to physical memory.

![Process Memory Model](./extracted-002.png)

Key points:
- Code, data, heap, and stack live in virtual memory.
- Multiple processes can map to the same physical memory (e.g. shared libraries).

---

#### 2.1.3 Process vs Thread

Threads **share most memory** within a process but have **independent stacks**.

![Thread Memory Layout](./extracted-003.png)

Summary:
- Heap, global data, and code are shared among threads.
- Each thread owns its own stack.

---

#### 2.1.4 Program’s Lifeline

How a C++ program runs from start to end:

![Program Lifeline](./extracted-004.png)

Steps:
1. Compilation & Linking
2. Loading into memory
3. Execution (starting from `main()`)
4. Runtime operations (system calls, exceptions)
5. Termination
6. Cleanup

---

## 2.2 Loading and Memory Allocation

### Loader Responsibilities

- Read executable from disk
- Allocate memory
- Load code and data into RAM

### Memory Segments

- **Text (Code)**: executable instructions
- **Data**: global & static variables
- **Stack**: function calls and local variables
- **Heap**: dynamic memory allocation

---

## 3. C++ Object-Oriented Memory Layout

### 3.1 Object Memory Model

Different parts of a C++ object live in different memory segments depending on how they are declared and created.
![OOP Object Code](./extracted-005.png)
![OOP Object Memory](./extracted-006.png)

Key observations:
- Member functions live in the text segment.
- Static members live in the data segment.
- Object instances live on stack or heap.

---

### 3.2 Inheritance Memory Model

Derived classes extend base-class memory layout.
![Inheritance Code](./extracted-007.png)
![Inheritance Memory Layout](./extracted-008.png)

#### Object Slicing

When a derived object is copied into a base object, derived parts are lost.
![Object Slicing Code](./extracted-009.png)
![Object Slicing Memory](./extracted-010.png)

Rule of thumb:
> Pass polymorphic objects by **reference or pointer**, never by value.

---

### 3.3 Virtual Function Memory Model (Polymorphism)

C++ implements polymorphism using **VTables**.
![Virtual Function Code](./extracted-011.png)
![Virtual Function Memory](./extracted-012.png)

Key ideas:
- Each class with virtual functions has a VTable.
- Each object contains a hidden `vptr` pointing to its class’s VTable.
- Virtual dispatch is resolved at runtime via the VTable.

---

## 4. Smart Pointer Memory Model

Smart pointers manage object lifetime automatically.

---

### 4.1 `unique_ptr`

- Exclusive ownership
- Zero overhead abstraction

![unique_ptr Memory](./extracted-013.png)

When the `unique_ptr` goes out of scope, the object is destroyed automatically.

---

### 4.2 `shared_ptr`

- Shared ownership
- Reference counting via control block

![shared_ptr Memory](./extracted-014.png)

Important:
- Object is destroyed when **reference count reaches zero**.
- Control block may outlive the object.

---

### 4.3 `weak_ptr`

- Non-owning observer of a `shared_ptr`
- Breaks reference cycles

![weak_ptr Memory](./extracted-015.png)

Rule:
> Use `weak_ptr` to observe, not to own.

---

## Final Notes

This memory perspective helps:
- Debug memory issues faster
- Design safer APIs
- Reason correctly about object lifetime
- Avoid common pitfalls in modern C++

If you truly understand **where things live**, you write better C++.

---

Happy hacking 🚀

