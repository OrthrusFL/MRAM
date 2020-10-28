package org.codeontology.neo4j;

import org.apache.commons.collections.CollectionUtils;
import org.codeontology.CodeOntology;
import org.codeontology.extraction.EntityFactory;
import org.codeontology.extraction.bgontology.MethodEntity;
import org.codeontology.extraction.bgontology.TypeEntity;
import org.codeontology.extraction.bgontology.TypeKind;
import org.joda.time.format.PeriodFormatter;
import org.joda.time.format.PeriodFormatterBuilder;
import spoon.processing.AbstractProcessor;
import spoon.reflect.declaration.CtType;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class AddCommentAndAPISeq extends AbstractProcessor<CtType<?>> {

    private String commitID;
    private List<String> updateMethodUriList;
    public static PeriodFormatter formatter = new PeriodFormatterBuilder().appendHours().appendSuffix(" h ").appendMinutes().appendSuffix(" min ").appendSeconds().appendSuffix(" s ").appendMillis().appendSuffix(" ms").toFormatter();

    public AddCommentAndAPISeq(String commitID, List<String> updateMethodUriList){
        this.commitID = commitID;
        this.updateMethodUriList = updateMethodUriList;
    }

    public static List<String> getUpdateMethodUriList(String commitID){
        String statement = String.format("match (n:Method) where n.commitID = '%s' return n.uri as methods", commitID);
        List<Object> methods = Neo4jQuery.cypherQuery(statement).get("methods");
        return methods.stream().map(n -> (String) n).collect(Collectors.toList());
    }

    @Override
    public void process(CtType<?> type) {
        if(TypeKind.getIgnoreTypeKind().contains(TypeKind.getKindOf(type))){
            return;
        }

        String className = type.getQualifiedName();
        TypeEntity typeEntity = EntityFactory.getInstance().wrap(type);

        List<MethodEntity> methodEntityList = typeEntity.getMethods();
        if(TypeKind.getKindOf(type).equals(TypeKind.CLASS)){
            methodEntityList.addAll(typeEntity.getConstructors());
        }

        if(Integer.valueOf(CodeOntology.getVersion()) > 1){//是版本更新
            methodEntityList = methodEntityList.stream().filter(m -> updateMethodUriList.contains(m.getNameVersion())).collect(Collectors.toList());
        }

        for(MethodEntity methodEntity : methodEntityList){
            if(!methodEntity.isDeclarationAvailable()){
                continue;
            }
            //add comment
            methodEntity.tagAllComment();
            //add api seq
            methodEntity.tagAPISeq();
        }
    }
}
