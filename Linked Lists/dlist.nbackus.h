#define MAX_STATE_NAME_LENGTH 64

typedef struct StateDListNodeStruct {
    int population;
    char name[MAX_STATE_NAME_LENGTH];
    struct StateDListNodeStruct *next;
    struct StateDListNodeStruct *prev;
} StateDListNode;