#define MAX_STATE_NAME_LENGTH 64

typedef struct StateListNodeStruct {
    int population;
    char name[MAX_STATE_NAME_LENGTH];
    struct StateListNodeStruct *next;
} StateListNode;
