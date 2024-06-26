TARGET_EXEC := cnn

BUILD_DIR := ./build
SRC_DIRS := ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c)


OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)


DEPS := $(OBJS:.o=.d)


INC_DIRS := $(shell find $(SRC_DIRS) -type d)

INC_FLAGS := $(addprefix -I,$(INC_DIRS))


CPPFLAGS := $(INC_FLAGS) -MMD -MP -g -Wall -O3 -fopenmp -pthread
LDFLAGS := -lm -lz -fopenmp -latomic

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)


$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@




.PHONY: clean
clean:
	rm -r $(BUILD_DIR)


-include $(DEPS)