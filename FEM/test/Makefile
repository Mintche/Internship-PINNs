CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -std=c++17 -I../include

SRC_DIR = ../src
OBJ_DIR = obj

# Objets de l'application nécessaires pour les tests
# On compile les sources de src/ (hors main) pour les lier aux tests
APP_OBJS = $(OBJ_DIR)/fem.o $(OBJ_DIR)/mesh.o $(OBJ_DIR)/linear_sampling.o

# Cible par défaut
all: test_math.x test_mesh.x test_fem.x test_linear_sampling.x test_p2_interpolation.x

# Compilation des tests
# test_math est purement basé sur des templates (math.hpp), pas besoin de lier d'objets
test_math.x: test_math.cpp ../include/math.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

test_mesh.x: test_mesh.cpp $(OBJ_DIR)/mesh.o
	$(CXX) $(CXXFLAGS) $^ -o $@

test_fem.x: test_fem.cpp $(OBJ_DIR)/fem.o $(OBJ_DIR)/mesh.o
	$(CXX) $(CXXFLAGS) $^ -o $@

test_linear_sampling.x: test_linear_sampling.cpp $(APP_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

test_p2_interpolation.x: test_p2_interpolation.cpp $(OBJ_DIR)/mesh.o $(OBJ_DIR)/p2_interpolation.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# Règle pour compiler les fichiers sources de l'application en objets
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

cleanobj:
	rm -rf $(OBJ_DIR)

# Nettoyage
clean:
	rm -rf $(OBJ_DIR) *.x

.PHONY: all clean
