#ifndef XACC_QUANTUM_GRAPH_HPP_
#define XACC_QUANTUM_GRAPH_HPP_

#include "Identifiable.hpp"
#include "variant.hpp"

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>

namespace exatn {

// Helper functions to print various types
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::pair<T, T> &p) {
  os << "[" << p.first << "," << p.second << "]";
  return os;
}
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1)
      os << ",";
  }
  os << "]";
  return os;
}

// Variant subclasses mpark::variant to add a few extra method calls
template <typename... Types> class Variant : public mpark::variant<Types...> {

private:
  class ToStringVisitor {
  public:
    template <typename T> std::string operator()(const T &t) const {
      std::stringstream ss;
      ss << t;
      return ss.str();
    }
  };
  class IsArithmeticVisitor {
  public:
    template <typename T> bool operator()(const T &t) const {
      return std::is_arithmetic<T>::value;
    }
  };

  std::map<int, std::string> whichType{{0,"int"}, {1,"double"},
                                        {2,"string"},
                                        {3,"vector<pair<int>>"},
                                        {4,"vector<pair<double>>"},
                                        {5,"vector<int>"},
                                        {6,"vector<double>"},
                                        {7,"vector<string>"}};
public:
  Variant() : mpark::variant<Types...>() {}
  template <typename T>
  Variant(T &element) : mpark::variant<Types...>(element) {}
  template <typename T>
  Variant(T &&element) : mpark::variant<Types...>(element) {}
  Variant(const Variant &element) : mpark::variant<Types...>(element) {}

  template <typename T> T as() const {
    try {
      // First off just try to get it
      return mpark::get<T>(*this);
    } catch (std::exception &e) {
        std::stringstream s;
        s << "This Variant type id is " << this->which() << "\nAllowed Ids to Type\n";
        for (auto& kv : whichType) {
            s << kv.first << ": " << kv.second << "\n";
        }
        std::cerr << "Cannot cast Variant:\n" << s.str() << "\n";
    }
    return T();
  }
  int which() const {
      return this->index();
  }
  bool isNumeric() const {
    IsArithmeticVisitor v;
    return mpark::visit(v, *this);
  }

  bool isVariable() const {
    try {
      mpark::get<std::string>(*this);
    } catch (std::exception &e) {
      return false;
    }
    return true;
  }

  const std::string toString() const {
    ToStringVisitor vis;
    return mpark::visit(vis, *this);
  }

  bool operator==(const Variant<Types...> &v) const {
    return v.toString() == toString();
  }

  bool operator!=(const Variant<Types...> &v) const { return !operator==(v); }

};

// Create the VertexProperty typedef
using VertexProperty =
    Variant<int, double, std::string,
            std::vector<std::pair<int, int>>,
            std::vector<std::pair<double, double>>,
            std::vector<int>, std::vector<double>, std::vector<std::string>>;

// Create the VertexProperties typedef, a map of strings to VertexProperties
using VertexProperties = std::map<std::string, VertexProperty>;

// Public Graph API
class Graph : public Identifiable, public Cloneable<Graph> {
public:

  virtual std::shared_ptr<Graph> clone() = 0;

  virtual void addEdge(const int srcIndex, const int tgtIndex,
               const double edgeWeight) = 0;
  virtual void addEdge(const int srcIndex, const int tgtIndex) = 0;
  virtual void removeEdge(const int srcIndex, const int tgtIndex) = 0;

  virtual void addVertex() = 0;
  virtual void addVertex(VertexProperties& properties) = 0;
  virtual void addVertex(VertexProperties&& properties) = 0;

  virtual void setVertexProperties(const int index, VertexProperties& properties) = 0;
  virtual void setVertexProperties(const int index, VertexProperties&& properties) = 0;
  virtual void setVertexProperty(const int index, const std::string prop, VertexProperty& p) = 0;
  virtual void setVertexProperty(const int index, const std::string prop, VertexProperty&& p) = 0;

  virtual VertexProperties getVertexProperties(const int index) = 0;
  virtual VertexProperty&
  getVertexProperty(const int index, const std::string property) = 0;

  virtual void setEdgeWeight(const int srcIndex, const int tgtIndex,
                     const double weight) = 0;
  virtual double getEdgeWeight(const int srcIndex, const int tgtIndex) = 0;
  virtual bool edgeExists(const int srcIndex, const int tgtIndex) = 0;

  virtual int degree(const int index) = 0;
  virtual int diameter()  = 0;
  // n edges
  virtual int size() = 0;
  // n vertices
  virtual int order() = 0;

  virtual const int depth() = 0;

  virtual std::vector<int> getNeighborList(const int index)  = 0;

  virtual void write(std::ostream &stream) = 0;
  virtual void read(std::istream &stream) = 0;

  virtual void computeShortestPath(int startIndex, std::vector<double> &distances,
                           std::vector<int> &paths) = 0;
};

}
#endif